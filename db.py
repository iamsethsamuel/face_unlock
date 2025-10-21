import sqlite3
from typing import Optional, List, Tuple
import os
import numpy as np
from utils import emb_to_blob, blob_to_emb

DB_SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    label TEXT
);

-- We store each embedding as a BLOB (float32 bytes) and the length
CREATE TABLE IF NOT EXISTS embeddings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER NOT NULL,
    embedding BLOB NOT NULL,
    dim INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(user_id) REFERENCES users(id) ON DELETE CASCADE
);
"""


def connect(db_path: str = "face_store.sqlite3") -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.row_factory = sqlite3.Row
    return conn


def init_db(conn: sqlite3.Connection):
    with conn:
        conn.executescript(DB_SCHEMA)


# demo: create DB, add user, store embeddings, verify
db_path = "demo_face_store.sqlite3"
if os.path.exists(db_path):
    os.remove(db_path)

conn = connect(db_path)
init_db(conn)


def add_user(conn: sqlite3.Connection, username: str, label: Optional[str] = None) -> int:
    with conn:
        cur = conn.execute(
            "INSERT INTO users (username, label) VALUES (?, ?)",
            (username, label)
        )
    return cur.lastrowid


def get_user_by_username(conn: sqlite3.Connection, username: str) -> Optional[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM users WHERE username = ?", (username,))
    return cur.fetchone()


def list_users(conn: sqlite3.Connection, limit: int = 100) -> List[sqlite3.Row]:
    cur = conn.execute("SELECT * FROM users ORDER BY id")
    return cur.fetchmany(size=limit)


def delete_user(conn: sqlite3.Connection, username: str):
    with conn:
        conn.execute("DELETE FROM users WHERE username = ?", (username,))


def add_embedding(conn: sqlite3.Connection, user_id: int, emb: np.ndarray):
    blob = emb_to_blob(emb)
    dim = int(np.asarray(emb).shape[0])
    with conn:
        conn.execute(
            "INSERT INTO embeddings (user_id, embedding, dim) VALUES (?, ?, ?)",
            (user_id, sqlite3.Binary(blob), dim)
        )


def get_embeddings_for_user(conn: sqlite3.Connection, user_id: int) -> List[np.ndarray]:
    cur = conn.execute("SELECT embedding, dim FROM embeddings WHERE user_id = ?", (user_id,))
    rows = cur.fetchall()
    return [blob_to_emb(row["embedding"], row["dim"]) for row in rows]


def get_all_embeddings(conn: sqlite3.Connection) -> List[Tuple[int, np.ndarray]]:
    """
    Returns a list of (user_id, embedding)
    """
    cur = conn.execute("SELECT user_id, embedding, dim FROM embeddings")
    rows = cur.fetchall()
    result = []
    for r in rows:
        emb = blob_to_emb(r["embedding"], r["dim"])
        result.append((r["user_id"], emb))
    return result

add_user(conn, "Seth", "The owner")
# print(list_users(conn)[0]["username"])

# user_rows = list_users(conn)
# print(f"Original output: {user_rows}\n")
# # Original output: [<sqlite3.Row object at 0x...>]
#
# # --- Here's how to get the data out ---
# for row in user_rows:
#     # Access by index
#     user_id_by_index = row[0]
#     user_name_by_index = row[1]
#     print(f"Accessed by index: ID={user_id_by_index}, Name={user_name_by_index}")
#
#     # Access by column name (more readable!)
#     user_id_by_name = row['id']
#     user_name_by_name = row['username']
#     print(f"Accessed by name:   ID={user_id_by_name}, Name={user_name_by_name}")
#
# conn.close()
