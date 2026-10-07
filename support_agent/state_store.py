import os
import sqlite3
import uuid
from typing import Any, Dict, List, Optional


class StateStore:
    """SQLite-backed state store for users, sessions, tickets, and order data."""

    def __init__(self, db_path: Optional[str] = None):
        base_dir = os.path.dirname(__file__)
        runtime_dir = os.path.join(base_dir, "runtime")
        os.makedirs(runtime_dir, exist_ok=True)
        self.db_path = db_path or os.path.join(runtime_dir, "support_agent.db")
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    display_name TEXT,
                    email TEXT,
                    tier TEXT DEFAULT 'standard',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS session_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS tickets (
                    ticket_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    session_id TEXT,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id),
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE TABLE IF NOT EXISTS ticket_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ticket_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(ticket_id) REFERENCES tickets(ticket_id)
                );

                CREATE TABLE IF NOT EXISTS orders (
                    order_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    amount TEXT NOT NULL,
                    items TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS shipments (
                    order_id TEXT PRIMARY KEY,
                    carrier TEXT NOT NULL,
                    tracking_no TEXT NOT NULL,
                    shipping_status TEXT NOT NULL,
                    eta TEXT NOT NULL,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(order_id) REFERENCES orders(order_id)
                );

                CREATE TABLE IF NOT EXISTS traces (
                    trace_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    query TEXT NOT NULL,
                    answer TEXT,
                    confidence REAL DEFAULT 0,
                    escalated INTEGER DEFAULT 0,
                    tool_name TEXT,
                    guardrail_reason TEXT,
                    failure_reason TEXT,
                    duration_ms INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    finished_at TEXT,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id),
                    FOREIGN KEY(user_id) REFERENCES users(user_id)
                );

                CREATE TABLE IF NOT EXISTS trace_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    trace_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(trace_id) REFERENCES traces(trace_id)
                );
                """
            )
            self._ensure_column(connection, "traces", "failure_reason", "TEXT")
        self.ensure_seed_data()

    def _ensure_column(
        self,
        connection: sqlite3.Connection,
        table_name: str,
        column_name: str,
        column_type: str,
    ) -> None:
        rows = connection.execute(f"PRAGMA table_info({table_name})").fetchall()
        existing_columns = {row["name"] for row in rows}
        if column_name not in existing_columns:
            connection.execute(
                f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
            )

    def ensure_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        tier: str = "standard",
    ) -> Dict[str, Any]:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users (user_id, display_name, email, tier)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(user_id) DO NOTHING
                """,
                (user_id, display_name, email, tier),
            )
        return self.get_user(user_id)

    def update_user(
        self,
        user_id: str,
        display_name: Optional[str] = None,
        email: Optional[str] = None,
        tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        current = self.ensure_user(user_id)
        next_display_name = display_name if display_name is not None else current.get("display_name")
        next_email = email if email is not None else current.get("email")
        next_tier = tier if tier is not None else current.get("tier", "standard")
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE users
                SET display_name = ?, email = ?, tier = ?, updated_at = CURRENT_TIMESTAMP
                WHERE user_id = ?
                """,
                (next_display_name, next_email, next_tier, user_id),
            )
        return self.get_user(user_id)

    def get_user(self, user_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        if not row:
            raise KeyError(f"User {user_id} was not found.")
        return dict(row)

    def create_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        self.ensure_user(user_id)
        session_id = session_id or f"session-{uuid.uuid4().hex[:12]}"
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO sessions (session_id, user_id)
                VALUES (?, ?)
                """,
                (session_id, user_id),
            )
        return self.get_session(session_id)

    def ensure_session(self, user_id: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        if session_id:
            try:
                session = self.get_session(session_id)
            except KeyError:
                session = self.create_session(user_id=user_id, session_id=session_id)
            if session["user_id"] != user_id:
                raise ValueError(
                    f"Session {session_id} belongs to user {session['user_id']}, not {user_id}."
                )
            return session
        return self.create_session(user_id=user_id)

    def append_session_message(self, session_id: str, role: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO session_messages (session_id, role, message)
                VALUES (?, ?, ?)
                """,
                (session_id, role, message),
            )
            connection.execute(
                """
                UPDATE sessions
                SET updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (session_id,),
            )

    def get_session(self, session_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            session_row = connection.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not session_row:
                raise KeyError(f"Session {session_id} was not found.")
            message_rows = connection.execute(
                """
                SELECT role, message, created_at
                FROM session_messages
                WHERE session_id = ?
                ORDER BY message_id ASC
                """,
                (session_id,),
            ).fetchall()
        session = dict(session_row)
        session["messages"] = [dict(row) for row in message_rows]
        return session

    def list_sessions(self, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query = "SELECT * FROM sessions"
        params: tuple[Any, ...] = ()
        if user_id:
            query += " WHERE user_id = ?"
            params = (user_id,)
        query += " ORDER BY updated_at DESC, created_at DESC"
        with self._connect() as connection:
            rows = connection.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def create_trace(self, trace_id: str, session_id: str, user_id: str, query: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO traces (trace_id, session_id, user_id, query)
                VALUES (?, ?, ?, ?)
                """,
                (trace_id, session_id, user_id, query),
            )

    def append_trace_event(self, trace_id: str, event_type: str, payload: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO trace_events (trace_id, event_type, payload)
                VALUES (?, ?, ?)
                """,
                (trace_id, event_type, payload),
            )

    def finish_trace(
        self,
        trace_id: str,
        answer: str,
        confidence: float,
        escalated: bool,
        tool_name: Optional[str],
        guardrail_reason: Optional[str],
        failure_reason: Optional[str],
        duration_ms: int,
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE traces
                SET answer = ?,
                    confidence = ?,
                    escalated = ?,
                    tool_name = ?,
                    guardrail_reason = ?,
                    failure_reason = ?,
                    duration_ms = ?,
                    finished_at = CURRENT_TIMESTAMP
                WHERE trace_id = ?
                """,
                (
                    answer,
                    confidence,
                    1 if escalated else 0,
                    tool_name,
                    guardrail_reason,
                    failure_reason,
                    duration_ms,
                    trace_id,
                ),
            )

    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        with self._connect() as connection:
            trace_row = connection.execute(
                "SELECT * FROM traces WHERE trace_id = ?",
                (trace_id,),
            ).fetchone()
            if not trace_row:
                raise KeyError(f"Trace {trace_id} was not found.")
            event_rows = connection.execute(
                """
                SELECT event_type, payload, created_at
                FROM trace_events
                WHERE trace_id = ?
                ORDER BY event_id ASC
                """,
                (trace_id,),
            ).fetchall()
        trace = dict(trace_row)
        trace["events"] = [dict(row) for row in event_rows]
        return trace

    def list_traces(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        query = "SELECT * FROM traces"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def create_ticket(
        self,
        user_id: str,
        initial_message: str,
        session_id: Optional[str] = None,
    ) -> int:
        self.ensure_user(user_id)
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO tickets (user_id, session_id)
                VALUES (?, ?)
                """,
                (user_id, session_id),
            )
            ticket_id = int(cursor.lastrowid)
            connection.execute(
                """
                INSERT INTO ticket_messages (ticket_id, role, message)
                VALUES (?, ?, ?)
                """,
                (ticket_id, "user", initial_message),
            )
        return ticket_id

    def add_ticket_message(self, ticket_id: int, role: str, message: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO ticket_messages (ticket_id, role, message)
                VALUES (?, ?, ?)
                """,
                (ticket_id, role, message),
            )
            connection.execute(
                """
                UPDATE tickets
                SET updated_at = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (ticket_id,),
            )

    def update_ticket_status(self, ticket_id: int, status: str) -> Dict[str, Any]:
        if status not in {"open", "pending", "closed"}:
            raise ValueError("status must be one of: closed, open, pending")
        with self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tickets
                SET status = ?, updated_at = CURRENT_TIMESTAMP
                WHERE ticket_id = ?
                """,
                (status, ticket_id),
            )
            if cursor.rowcount == 0:
                raise KeyError(f"Ticket {ticket_id} was not found.")
        return self.get_ticket(ticket_id)

    def get_ticket(self, ticket_id: int) -> Dict[str, Any]:
        with self._connect() as connection:
            ticket_row = connection.execute(
                "SELECT * FROM tickets WHERE ticket_id = ?",
                (ticket_id,),
            ).fetchone()
            if not ticket_row:
                raise KeyError(f"Ticket {ticket_id} was not found.")
            message_rows = connection.execute(
                """
                SELECT role, message, created_at
                FROM ticket_messages
                WHERE ticket_id = ?
                ORDER BY message_id ASC
                """,
                (ticket_id,),
            ).fetchall()
        ticket = dict(ticket_row)
        ticket["messages"] = [dict(row) for row in message_rows]
        return ticket

    def list_tickets(
        self,
        status: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        clauses: List[str] = []
        params: List[Any] = []
        if status:
            clauses.append("status = ?")
            params.append(status)
        if user_id:
            clauses.append("user_id = ?")
            params.append(user_id)
        query = "SELECT * FROM tickets"
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY updated_at DESC, ticket_id DESC"
        with self._connect() as connection:
            rows = connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_order(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM orders WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        return dict(row) if row else None

    def get_shipment(self, order_id: str) -> Optional[Dict[str, Any]]:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM shipments WHERE order_id = ?",
                (order_id,),
            ).fetchone()
        return dict(row) if row else None

    def ensure_seed_data(self) -> None:
        self.ensure_user("user-1", display_name="User One", tier="vip")
        self.ensure_user("user-2", display_name="User Two")
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO orders (order_id, user_id, status, amount, items)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("ORD-1001", "user-1", "paid", "HKD 299", "wireless headset"),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO orders (order_id, user_id, status, amount, items)
                VALUES (?, ?, ?, ?, ?)
                """,
                ("ORD-1002", "user-2", "shipped", "HKD 88", "phone case"),
            )
            connection.execute(
                """
                INSERT OR IGNORE INTO shipments (
                    order_id, carrier, tracking_no, shipping_status, eta
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    "ORD-1002",
                    "SF Express",
                    "SF123456789HK",
                    "in_transit",
                    "2026-06-15",
                ),
            )
