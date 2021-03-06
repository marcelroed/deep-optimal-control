import subprocess
import pathlib as p
from typing import Optional
from tqdm import tqdm


def render_markdown(file_path: p.Path):
    """Render markdown to PDF, use same stem for filename."""
    arguments = ['pandoc', str(file_path), '-o',
                 str(file_path.parents[0]) + '\\' + "_" + str(file_path.stem) + '.pdf']
    subprocess.call(arguments)


def find_with_ext(top_dir: p.Path, ext: str):
    """Return all files in subdirectories with a certain extension"""
    assert len(ext) > 0, 'File extension cannot be an empty string.'
    if ext[0] != '.':
        ext = '.' + ext

    md_files = []
    for sub_path in top_dir.glob('**/*'):
        file_ext = sub_path.suffix
        if file_ext == ext:
            md_files.append(sub_path)
    return md_files


def get_pdf(file_path: p.Path) -> Optional[p.Path]:
    """Return associated pdf if it exists, else return None"""
    for file_in_folder in file_path.parent.glob('*'):
        # Look through local directory
        if file_in_folder.stem == '_' + file_path.stem and file_in_folder.suffix == '.pdf':
            return file_in_folder
    return None


def is_older(file_1: p.Path, file_2: p.Path) -> bool:
    """Return true iff file_1 is older than file_2"""
    return file_1.stat().st_mtime < file_2.stat().st_mtime


def render_recursively(path: p.Path) -> None:
    # Paths of all Markdown files where their PDF either does not exist or is older than the Markdown file.
    to_render = [p for p, pdf in
                 ((md_path, get_pdf(md_path)) for md_path in find_with_ext(path, '.md')) if
                 pdf is None or is_older(pdf, p)]

    # Render Markdown files
    for markdown_path in tqdm(to_render, desc='Rendering markdown files', unit='files'):
        # PDF either does not exist, or the one that does is older than the newest markdown file
        render_markdown(markdown_path)


if __name__ == '__main__':
    render_recursively(p.Path('../'))
