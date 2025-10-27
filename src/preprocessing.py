import sys, re, csv
import pandas as pd

try:
    csv.field_size_limit(sys.maxsize)
except OverflowError:
    csv.field_size_limit(2**31 - 1)

df = pd.read_csv(
    'song_lyrics.csv',
    sep=',',
    quotechar='"',
    quoting=csv.QUOTE_MINIMAL,
    doublequote=True,
    engine='python',
    encoding='utf-8',
    on_bad_lines='skip'
)

cols = ['title', 'artist', 'tag', 'lyrics']
df_keep = df.loc[:, cols].copy()
total = len(df_keep)

def clean_lyrics(s: str) -> str:
    if not isinstance(s, str): return ""
    s = re.sub(r'\[[^\n\]]*?\]', '', s)
    s = re.sub(r'\([^\n\)]*?\)', '', s)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r' *\n *', '\n', s)
    s = re.sub(r'\n{3,}', '\n\n', s)
    return s.strip()


cleaned = []
for i, s in enumerate(df_keep['lyrics'], start=1):
    cleaned.append(clean_lyrics(s))
    if i % 100 == 0:
        print(f'cleaned {i}/{total} rows')

df_keep['lyrics'] = cleaned


before_filter = len(df_keep)
df_keep = df_keep[df_keep['lyrics'].str.len().ge(20)]
after_len_filter = len(df_keep)
df_keep = df_keep.drop_duplicates(subset=['title', 'artist', 'lyrics'])
after_dedup = len(df_keep)

print(f'kept length>=20: {after_len_filter}/{before_filter}')
print(f'after dedup: {after_dedup}/{after_len_filter}')

out_path = 'clean-with-title-artist-all.csv'
df_keep.to_csv(out_path, index=False)
print('generated:', out_path, 'shape:', df_keep.shape)
