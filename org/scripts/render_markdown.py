import subprocess
import pathlib as p
import tqdm


def render_markdown(file_path: p.Path):
    """Render markdown to PDF, use same stem for filename."""
    arguments = ['pandoc', str(file_path), '--pdf-engine=latexmk', '-o', str(file_path.parents[0]) + '\\' + str(file_path.stem) + '.pdf']
    print(arguments)
    subprocess.call(arguments)


def find_with_ext(top_dir: p.Path, ext: str):
    """Return all files in subdirectories with a certain extension"""
    if ext[0] != '.':
        ext = '.' + ext

    md_files = []
    for sub_path in top_dir.glob('**/*'):
        file_ext = sub_path.suffix
        if file_ext == ext:
            md_files.append(sub_path)

    return md_files


if __name__ == '__main__':
    for file_path in tqdm.tqdm(find_with_ext(p.Path('../'), 'md')):
        render_markdown(file_path)
