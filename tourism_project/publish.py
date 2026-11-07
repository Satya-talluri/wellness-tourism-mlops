"""Explicitly authorized, single-commit publication from a notebook."""

import base64
import hashlib
import json
from pathlib import Path
import re

from tourism_project.schema import ROOT


def publish_project(token, repository_name, allow_public=False, root=ROOT):
    if not allow_public:
        print("Publication skipped: public upload is not authorized.")
        return None
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,99}", repository_name):
        raise ValueError("Enter a valid repository name, not a URL.")
    if not token:
        raise ValueError("A GitHub token is required.")

    from github import Auth, Github, GithubException, InputGitTreeElement

    root = Path(root)
    required = [
        ".gitignore",
        ".github/workflows/pipeline.yml",
        "requirements.txt",
        "tourism_project/__init__.py",
        "tourism_project/publish.py",
        "tourism_project/schema.py",
        "tourism_project/data/tourism.csv",
        "tourism_project/requirements.txt",
        "tourism_project/model_building/data_register.py",
        "tourism_project/model_building/prep.py",
        "tourism_project/model_building/train.py",
        "tourism_project/deployment/app.py",
        "tourism_project/deployment/requirements.txt",
        "tourism_project/deployment/model.joblib",
        "tourism_project/deployment/model_metadata.json",
        "tests/test_data.py",
        "tests/test_model_and_app.py",
    ]
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Finish the earlier sections first: {missing}")

    with Github(auth=Auth.Token(token)) as client:
        user = client.get_user()
        full_name = f"{user.login}/{repository_name}"
        try:
            repo = client.get_repo(full_name)
        except GithubException as error:
            if error.status != 404:
                raise
            repo = user.create_repo(repository_name, private=False, auto_init=True)
        if repo.private:
            raise ValueError(
                "This repository is private. Choose an authorized public repository."
            )
        if repo.default_branch != "main":
            raise ValueError(
                "This workflow targets main. Set the repository default branch "
                "to main before publishing."
            )
        branch = repo.get_branch("main")
        parent = repo.get_git_commit(branch.commit.sha)
        existing = {
            item.path: item.sha
            for item in repo.get_git_tree(parent.tree.sha, recursive=True).tree
            if item.type == "blob"
        }
        links_path = root / "project_links.json"
        links = json.loads(links_path.read_text()) if links_path.exists() else {}
        links["github_repository"] = repo.html_url
        links.setdefault("streamlit_app", "")
        links_path.write_text(json.dumps(links, indent=2) + "\n")
        paths = required + ["project_links.json"]
        elements = []
        for name in paths:
            content = (root / name).read_bytes()
            git_hash = hashlib.sha1(
                b"blob " + str(len(content)).encode() + b"\0" + content
            ).hexdigest()
            if existing.get(name) == git_hash:
                continue
            blob = repo.create_git_blob(
                base64.b64encode(content).decode("ascii"), "base64"
            )
            elements.append(
                InputGitTreeElement(path=name, mode="100644", type="blob", sha=blob.sha)
            )
        if elements:
            tree = repo.create_git_tree(elements, base_tree=parent.tree)
            commit = repo.create_git_commit(
                "Publish Wellness Tourism MLOps pipeline files",
                tree,
                [parent],
            )
            repo.get_git_ref("heads/main").edit(commit.sha, force=False)
            commit_sha = commit.sha
        else:
            commit_sha = parent.sha
            print("Repository already contains the same selected project files.")
        receipt = {
            "repository": repo.full_name,
            "github_url": repo.html_url,
            "commit_sha": commit_sha,
            "changed_files": len(elements),
        }
        reports = root / "reports"
        reports.mkdir(exist_ok=True)
        (reports / "github_publication.json").write_text(
            json.dumps(receipt, indent=2) + "\n"
        )
        print(json.dumps(receipt, indent=2))
        print("Publication is recorded; check Actions separately for execution status.")
        return receipt
