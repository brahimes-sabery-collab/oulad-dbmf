"""Download only the two required CSVs from the official UCI archive."""
from pathlib import Path
from urllib.request import urlopen
from io import BytesIO
from zipfile import ZipFile

URL = 'https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip'
if __name__ == '__main__':
    destination = Path('data/oulad')
    destination.mkdir(parents=True, exist_ok=True)
    with urlopen(URL, timeout=120) as response:
        archive = ZipFile(BytesIO(response.read(100_000_000)))
    for name in ('studentInfo.csv', 'studentRegistration.csv'):
        matches = [p for p in archive.namelist() if p.split('/')[-1] == name]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {name}')
        target = destination / name
        if target.exists():
            raise FileExistsError(f'Refusing to overwrite {target}')
        target.write_bytes(archive.read(matches[0]))
        print(f'Downloaded {target}')
