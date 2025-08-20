import os
import sqlite3
from contextlib import contextmanager

# Resolve path to SQLite DB (same location used in app.core.database)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(BASE_DIR, "legal_assistant.db")


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    try:
        yield conn
    finally:
        conn.close()


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (name,)
    )
    return cur.fetchone() is not None


def column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    cur = conn.execute(f"PRAGMA table_info('{table}')")
    return any(row[1] == column for row in cur.fetchall())


def sqlite_supports_drop_column(conn: sqlite3.Connection) -> bool:
    cur = conn.execute("select sqlite_version()")
    version = cur.fetchone()[0]
    major, minor, patch = (int(x) for x in version.split(".")[:3])
    # DROP COLUMN supported from 3.35.0
    return (major, minor, patch) >= (3, 35, 0)


def drop_legal_question_if_present(conn: sqlite3.Connection):
    if not table_exists(conn, "case_files"):
        print(
            "[MIGRATION] case_files table not found; skipping drop of legal_question."
        )
        return
    if not column_exists(conn, "case_files", "legal_question"):
        print("[MIGRATION] Column legal_question already absent.")
        return

    print("[MIGRATION] Dropping column legal_question from case_files...")
    if sqlite_supports_drop_column(conn):
        try:
            conn.execute("ALTER TABLE case_files DROP COLUMN legal_question")
            print("[MIGRATION] Column legal_question dropped via ALTER TABLE.")
            return
        except sqlite3.OperationalError as e:
            print(
                f"[MIGRATION] Direct DROP COLUMN failed ({e}); falling back to table rebuild."
            )

    # Fallback: table rebuild
    cur = conn.execute("PRAGMA foreign_keys")
    fk_initial = cur.fetchone()[0]
    conn.execute("PRAGMA foreign_keys=OFF")
    try:
        conn.execute("BEGIN")
        # Create new table without legal_question
        conn.execute(
            """CREATE TABLE case_files_new (
                id INTEGER PRIMARY KEY, 
                title VARCHAR(255) NOT NULL, 
                description TEXT, 
                user_facts TEXT, 
                created_at DATETIME, 
                updated_at DATETIME
            )"""
        )
        # Copy data
        conn.execute(
            """INSERT INTO case_files_new (id, title, description, user_facts, created_at, updated_at)
               SELECT id, title, description, user_facts, created_at, updated_at FROM case_files"""
        )
        # Replace table
        conn.execute("DROP TABLE case_files")
        conn.execute("ALTER TABLE case_files_new RENAME TO case_files")
        conn.execute("COMMIT")
        print("[MIGRATION] Column legal_question removed via table rebuild.")
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        if fk_initial:
            conn.execute("PRAGMA foreign_keys=ON")


def create_moot_court_sessions_if_missing(conn: sqlite3.Connection):
    if table_exists(conn, "moot_court_sessions"):
        print("[MIGRATION] moot_court_sessions table already exists.")
        return

    print("[MIGRATION] Creating table moot_court_sessions...")
    conn.execute(
        """CREATE TABLE moot_court_sessions (
            id INTEGER PRIMARY KEY, 
            case_file_id INTEGER NOT NULL, 
            draft_id INTEGER, 
            title VARCHAR(255) NOT NULL, 
            counterarguments JSON NOT NULL, 
            rebuttals JSON NOT NULL, 
            source_arguments JSON, 
            research_context JSON, 
            counterargument_strength FLOAT, 
            research_comprehensiveness FLOAT, 
            rebuttal_quality FLOAT, 
            execution_time FLOAT, 
            created_at DATETIME, 
            FOREIGN KEY(case_file_id) REFERENCES case_files(id) ON DELETE CASCADE, 
            FOREIGN KEY(draft_id) REFERENCES argument_drafts(id) ON DELETE SET NULL
        )"""
    )
    print("[MIGRATION] Table moot_court_sessions created.")


def add_party_represented_if_missing(conn: sqlite3.Connection):
    if not table_exists(conn, "case_files"):
        print(
            "[MIGRATION] case_files table not found; skipping add of party_represented."
        )
        return
    if column_exists(conn, "case_files", "party_represented"):
        print("[MIGRATION] Column party_represented already exists.")
        return

    print("[MIGRATION] Adding column party_represented to case_files...")
    try:
        conn.execute("ALTER TABLE case_files ADD COLUMN party_represented VARCHAR(100)")
        print("[MIGRATION] Column party_represented added successfully.")
    except sqlite3.OperationalError as e:
        print(f"[MIGRATION] Failed to add party_represented column: {e}")
        raise


def create_case_file_notes_if_missing(conn: sqlite3.Connection):
    if table_exists(conn, "case_file_notes"):
        print("[MIGRATION] case_file_notes table already exists.")
        return

    print("[MIGRATION] Creating case_file_notes table...")
    try:
        conn.execute("""
            CREATE TABLE case_file_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                case_file_id INTEGER NOT NULL,
                content TEXT NOT NULL,
                author_type VARCHAR(20) NOT NULL,
                author_name VARCHAR(100),
                note_type VARCHAR(50),
                tags JSON,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at DATETIME,
                FOREIGN KEY (case_file_id) REFERENCES case_files (id) ON DELETE CASCADE
            )
        """)
        print("[MIGRATION] case_file_notes table created successfully.")
    except sqlite3.OperationalError as e:
        print(f"[MIGRATION] Failed to create case_file_notes table: {e}")
        raise


def migration_already_applied(conn: sqlite3.Connection) -> bool:
    no_legal_question = table_exists(conn, "case_files") and not column_exists(
        conn, "case_files", "legal_question"
    )
    moot_exists = table_exists(conn, "moot_court_sessions")
    party_exists = table_exists(conn, "case_files") and column_exists(
        conn, "case_files", "party_represented"
    )
    notes_exists = table_exists(conn, "case_file_notes")
    return no_legal_question and moot_exists and party_exists and notes_exists


def apply_migration():
    if not os.path.exists(DB_PATH):
        print(
            f"[MIGRATION] Database not found at {DB_PATH}. Creating fresh schema via application startup will suffice."
        )
        return

    with get_conn() as conn:
        if migration_already_applied(conn):
            print("[MIGRATION] Schema already up to date. No action needed.")
            return

        print("[MIGRATION] Starting database schema migration...")
        drop_legal_question_if_present(conn)
        create_moot_court_sessions_if_missing(conn)
        add_party_represented_if_missing(conn)
        create_case_file_notes_if_missing(conn)
        conn.commit()
        print("[MIGRATION] Migration complete.")


if __name__ == "__main__":
    try:
        apply_migration()
    except Exception as e:
        print(f"[MIGRATION] Migration failed: {e}")
        raise
