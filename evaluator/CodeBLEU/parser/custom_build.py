from git import Git, Repo
from pathlib import Path
from tree_sitter import Language

_GRAMMARs = {
    #"go": ("https://github.com/tree-sitter/tree-sitter-go.git", "tree-sitter-go", "v0.13.3"),
    "java": ("https://github.com/tree-sitter/tree-sitter-java.git", "tree-sitter-java", "v0.19.0"),
    #"javascript": ("https://github.com/tree-sitter/tree-sitter-javascript.git", "tree-sitter-javascript", "v0.13.10"),
    #"python": ("https://github.com/tree-sitter/tree-sitter-python.git", "tree-sitter-python", "v0.14.0"),
    #"ruby": ("https://github.com/tree-sitter/tree-sitter-ruby.git", "tree-sitter-ruby", "v0.19.0"),
    "c-sharp": ("https://github.com/tree-sitter/tree-sitter-c-sharp.git", "tree-sitter-c-sharp", "v0.19.1"),
}

# if __name__ == '__main__':
def main():
    languages = []
    for lang, (url, dir, tag) in _GRAMMARs.items():
        #repo_dir = Path(function_parser.__path__[0])/dir
        repo_dir = Path(dir)
        if not repo_dir.exists():
            repo = Repo.clone_from(url, repo_dir)
        g = Git(str(repo_dir))
        g.checkout(tag)
        languages.append(str(repo_dir))
    
    Language.build_library(
        # Store the library in the directory
        "my-languages.so",
        # Include one or more languages
        languages
    )

if __name__ == "__main__":
    main()
