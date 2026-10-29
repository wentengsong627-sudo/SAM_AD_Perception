"""Install only the pinned FlyCo perception adapter, with preflight and backups.

Python 3.8+; no ROS, Torch or Internet is needed for installation.
"""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import uuid

HERE=Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data.replace(b'\r\n',b'\n')).hexdigest()


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args]).decode('utf-8').strip()


def inside(root,relative):
    path=root/relative
    path.resolve().relative_to(root.resolve())
    return path


def reconstruct(repo,package,item,commit):
    relative=item['path']
    if item['before_sha256'] is None:
        return inside(package/'new_files',relative).read_bytes()
    raw=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+relative])
    if digest(raw)!=item['before_sha256']:
        raise RuntimeError('Unexpected base source: '+relative)
    lines=raw.decode('utf-8').replace('\r\n','\n').splitlines(True)
    edits=json.loads((package/'edits.json').read_text(encoding='utf-8'))[relative]
    for edit in reversed(edits):
        lines[edit['start']:edit['end']]=edit['insert']
    return ''.join(lines).encode('utf-8')


def inspect(repo,package=HERE):
    repo=Path(repo).resolve(); package=Path(package).resolve()
    manifest=json.loads((package/'manifest.json').read_text(encoding='utf-8'))
    actual=git(repo,'rev-parse','HEAD')
    if actual!=manifest['base_commit']:
        raise RuntimeError('Unsupported FlyCo commit '+actual+'; expected '+manifest['base_commit']+'. No files were changed.')
    targets={row['path'] for row in manifest['files']}
    dirty=set(git(repo,'diff','--name-only','HEAD').splitlines())
    protected_dirty=sorted(dirty-targets)
    if protected_dirty:
        raise RuntimeError('Changes outside the adapter scope already exist: '+str(protected_dirty[:10])+'. Left untouched.')
    plan=[]
    for item in manifest['files']:
        relative=item['path']; target=inside(repo,relative); source=reconstruct(repo,package,item,manifest['base_commit'])
        if digest(source)!=item['after_sha256']:
            raise RuntimeError('Plugin integrity failure: '+relative)
        if target.is_symlink():raise RuntimeError('Refusing a symlink target: '+relative)
        before=target.read_bytes() if target.is_file() else None
        current=None if before is None else digest(before)
        if current==item['after_sha256']:action='already_installed'
        elif current==item['before_sha256']:action='install'
        else:raise RuntimeError('Local edits or unexpected file: '+relative+'. No files were changed.')
        plan.append((item,target,source,before,action))
    return manifest,plan


def install(repo,check=False,package=HERE):
    repo=Path(repo).resolve(); package=Path(package).resolve()
    manifest,plan=inspect(repo,package)
    pending=[row for row in plan if row[4]=='install']
    report={'base_commit':manifest['base_commit'],'repo':str(repo),'checked_files':len(plan),
            'files_to_install':len(pending),'outside_perception_backend_changes':False,
            'dry_run':bool(check),'backup':None}
    if check or not pending:return report
    name=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    backup=repo/'.sam-ad-backups'/name
    backup.mkdir(parents=True,exist_ok=False)
    # Save all affected existing files before writing any replacements.
    for item,target,source,before,_ in pending:
        if before is not None:
            destination=inside(backup,item['path']); destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(target,destination)
    (backup/'manifest.json').write_text(json.dumps({'base_commit':manifest['base_commit'],
        'files':[dict(row[0],existed=row[3] is not None) for row in pending]},indent=2),encoding='utf-8')
    written=[]
    try:
        for item,target,source,before,_ in pending:
            current=target.read_bytes() if target.is_file() else None
            if current!=before:raise RuntimeError('File changed during installation: '+item['path'])
            target.parent.mkdir(parents=True,exist_ok=True)
            temporary=target.with_name(target.name+'.sam-ad-'+uuid.uuid4().hex+'.tmp')
            try:
                temporary.write_bytes(source)
                if target.exists():shutil.copymode(target,temporary)
                os.replace(str(temporary),str(target)); written.append((item,target,before))
            finally:
                if temporary.exists():temporary.unlink()
        inspect(repo,package)
    except Exception:
        for item,target,before in reversed(written):
            if digest(target.read_bytes())!=item['after_sha256']:
                raise RuntimeError('Concurrent edit prevents automatic rollback; originals remain in '+str(backup))
            if before is None:target.unlink()
            else:shutil.copy2(inside(backup,item['path']),target)
        raise
    report.update(backup=str(backup),installed_files=len(written))
    (backup/'installation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',type=Path,required=True,help='Existing FC-Family/FlyCo checkout')
    parser.add_argument('--check',action='store_true',help='Read-only preflight')
    args=parser.parse_args()
    print(json.dumps(install(args.repo,args.check),ensure_ascii=False,indent=2))


if __name__=='__main__':main()
