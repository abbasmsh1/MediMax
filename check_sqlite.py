import sqlite3

db = r'storage\chroma_db\chroma.sqlite3'
conn = sqlite3.connect(db)
cur = conn.cursor()

cur.execute('SELECT COUNT(*) FROM embeddings')
total = cur.fetchone()[0]
print(f'Total chunks: {total}')

cur.execute("""
    SELECT string_value as src, COUNT(*) as cnt
    FROM embedding_metadata
    WHERE key='source'
    GROUP BY src
    ORDER BY cnt DESC
    LIMIT 20
""")
rows = cur.fetchall()
print('Sources:')
for src, cnt in rows:
    print(f'  {cnt:6d}  {src}')

conn.close()
