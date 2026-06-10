"""Pre-built tool library — batteries-included tools for agents."""

from __future__ import annotations

from nexus.core.types import ToolParameter
from nexus.tools.library.calculator import calculator
from nexus.tools.library.code_analysis_tools import (
    analyze_file_tool,
    check_types_tool,
    find_symbol_tool,
    lint_code_tool,
    summarize_structure_tool,
)
from nexus.tools.library.document_tools import read_docx_tool, read_excel_tool, read_pdf_tool
from nexus.tools.library.document_types import (
    ChunkStrategy,
    DocumentChunk,
    DocumentMetadata,
    ParsedDocument,
)
from nexus.tools.library.file_read import file_read
from nexus.tools.library.file_write import file_write
from nexus.tools.library.http_request import http_request
from nexus.tools.library.send_email import send_email
from nexus.tools.library.sql_query import sql_query
from nexus.tools.library.web_search import web_search
from nexus.tools.registry import ToolRegistry

LIBRARY_REGISTRY: ToolRegistry = ToolRegistry()

LIBRARY_REGISTRY.register(
    read_pdf_tool,
    name="read_pdf",
    description=(
        "Parse a PDF file into structured text chunks ready for RAG or memory ingestion. "
        "Returns page numbers, heading paths, and token estimates per chunk. "
        "Requires: pip install 'nexus-ai[documents]'."
    ),
    parameters=[
        ToolParameter(
            name="path", type="string", description="Path to the .pdf file.", required=True
        ),
        ToolParameter(
            name="chunk_size",
            type="integer",
            description="Target tokens per chunk.",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="chunk_strategy",
            type="string",
            description="'recursive' or 'fixed'.",
            required=False,
            default="recursive",
            enum=["recursive", "fixed"],
        ),
    ],
)

LIBRARY_REGISTRY.register(
    read_docx_tool,
    name="read_docx",
    description=(
        "Parse a Word (.docx) file into structured text chunks with heading-based section paths. "
        "Requires: pip install 'nexus-ai[documents]'."
    ),
    parameters=[
        ToolParameter(
            name="path", type="string", description="Path to the .docx file.", required=True
        ),
        ToolParameter(
            name="chunk_size",
            type="integer",
            description="Target tokens per chunk.",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="chunk_strategy",
            type="string",
            description="'recursive' or 'fixed'.",
            required=False,
            default="recursive",
            enum=["recursive", "fixed"],
        ),
    ],
)

LIBRARY_REGISTRY.register(
    read_excel_tool,
    name="read_excel",
    description=(
        "Parse an Excel (.xlsx) file into structured text chunks, one chunk set per sheet. "
        "Rows rendered as Markdown tables (capped at 1000 rows/sheet). "
        "Requires: pip install 'nexus-ai[documents]'."
    ),
    parameters=[
        ToolParameter(
            name="path", type="string", description="Path to the .xlsx file.", required=True
        ),
        ToolParameter(
            name="chunk_size",
            type="integer",
            description="Target tokens per chunk.",
            required=False,
            default=512,
        ),
        ToolParameter(
            name="chunk_strategy",
            type="string",
            description="'recursive' or 'fixed'.",
            required=False,
            default="recursive",
            enum=["recursive", "fixed"],
        ),
    ],
)

LIBRARY_REGISTRY.register(
    calculator,
    name="calculator",
    description="Safely evaluate arithmetic expressions (no eval — AST-based). Supports +,-,*,/,**,%,sqrt,abs,round,floor,ceil,log,sin,cos,tan,pi,e.",
    parameters=[
        ToolParameter(
            name="expression",
            type="string",
            description='Math expression to evaluate, e.g. "sqrt(16) + 2 * pi".',
            required=True,
        )
    ],
)

LIBRARY_REGISTRY.register(
    http_request,
    name="http_request",
    description="Make an HTTP request and return the response body. Response body is truncated to 10,000 characters.",
    parameters=[
        ToolParameter(
            name="url",
            type="string",
            description="Target URL.",
            required=True,
        ),
        ToolParameter(
            name="method",
            type="string",
            description="HTTP method.",
            required=True,
            enum=["GET", "POST", "PUT", "PATCH", "DELETE"],
        ),
        ToolParameter(
            name="headers",
            type="object",
            description="Optional request headers as a key-value dict.",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="body",
            type="object",
            description="Optional JSON body for POST/PUT/PATCH.",
            required=False,
            default=None,
        ),
        ToolParameter(
            name="timeout_seconds",
            type="number",
            description="Request timeout in seconds.",
            required=False,
            default=10.0,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    file_read,
    name="file_read",
    description="Read a file from disk. Path must be within allowed_base_dir to prevent traversal attacks.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the file to read.",
            required=True,
        ),
        ToolParameter(
            name="allowed_base_dir",
            type="string",
            description="Root directory that restricts which files can be read.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="max_bytes",
            type="integer",
            description="Maximum bytes to read (default 102400 = 100 KB).",
            required=False,
            default=102400,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    file_write,
    name="file_write",
    description="Write text content to a file. Path must be within allowed_base_dir.",
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Destination file path.",
            required=True,
        ),
        ToolParameter(
            name="content",
            type="string",
            description="Text content to write.",
            required=True,
        ),
        ToolParameter(
            name="allowed_base_dir",
            type="string",
            description="Root directory that restricts which files can be written.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="create_dirs",
            type="boolean",
            description="Create parent directories if they don't exist.",
            required=False,
            default=True,
        ),
        ToolParameter(
            name="overwrite",
            type="boolean",
            description="Whether to overwrite an existing file.",
            required=False,
            default=True,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    web_search,
    name="web_search",
    description="Search the web via DuckDuckGo Instant Answer API (no API key required). Returns titles, URLs, and snippets.",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="Search query string.",
            required=True,
        ),
        ToolParameter(
            name="max_results",
            type="integer",
            description="Maximum number of results to return (capped at 10).",
            required=False,
            default=5,
        ),
        ToolParameter(
            name="region",
            type="string",
            description='DuckDuckGo region code, e.g. "us-en", "wt-wt" (worldwide).',
            required=False,
            default="wt-wt",
        ),
    ],
)

LIBRARY_REGISTRY.register(
    sql_query,
    name="sql_query",
    description="Execute a read-only SELECT query against a database. Requires nexus-ai[sql] (sqlalchemy + aiosqlite).",
    parameters=[
        ToolParameter(
            name="query",
            type="string",
            description="SQL SELECT statement to execute.",
            required=True,
        ),
        ToolParameter(
            name="connection_string",
            type="string",
            description='SQLAlchemy async connection string, e.g. "sqlite+aiosqlite:///db.sqlite3".',
            required=True,
        ),
        ToolParameter(
            name="max_rows",
            type="integer",
            description="Maximum number of rows to return.",
            required=False,
            default=100,
        ),
        ToolParameter(
            name="timeout_seconds",
            type="number",
            description="Query execution timeout in seconds.",
            required=False,
            default=10.0,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    send_email,
    name="send_email",
    description="Send a plain-text email via SMTP using stdlib smtplib.",
    parameters=[
        ToolParameter(
            name="to",
            type="string",
            description="Recipient email address.",
            required=True,
        ),
        ToolParameter(
            name="subject",
            type="string",
            description="Email subject line.",
            required=True,
        ),
        ToolParameter(
            name="body",
            type="string",
            description="Plain-text email body.",
            required=True,
        ),
        ToolParameter(
            name="smtp_host",
            type="string",
            description="SMTP server hostname.",
            required=False,
            default="localhost",
        ),
        ToolParameter(
            name="smtp_port",
            type="integer",
            description="SMTP server port.",
            required=False,
            default=587,
        ),
        ToolParameter(
            name="username",
            type="string",
            description="SMTP auth username (empty = no auth).",
            required=False,
            default="",
        ),
        ToolParameter(
            name="password",
            type="string",
            description="SMTP auth password.",
            required=False,
            default="",
        ),
        ToolParameter(
            name="from_address",
            type="string",
            description="Sender email address.",
            required=False,
            default="nexus@localhost",
        ),
        ToolParameter(
            name="use_tls",
            type="boolean",
            description="Whether to use STARTTLS.",
            required=False,
            default=True,
        ),
    ],
)


LIBRARY_REGISTRY.register(
    analyze_file_tool,
    name="analyze_file",
    description=(
        "Full structural analysis of a Python (.py) file: all functions with signatures, "
        "cyclomatic complexity, decorators, and line ranges; all classes with methods and "
        "bases; all imports categorized as stdlib/third-party/local. "
        "Zero dependencies — uses Python ast stdlib."
    ),
    parameters=[
        ToolParameter(
            name="path", type="string", description="Path to the .py file.", required=True
        ),
    ],
)

LIBRARY_REGISTRY.register(
    lint_code_tool,
    name="lint_code",
    description=(
        "Run Ruff linter on a Python file or directory. Returns structured findings with "
        "rule ID, message, line, column, and fix availability. "
        "Degrades gracefully if Ruff is not installed."
    ),
    parameters=[
        ToolParameter(
            name="path",
            type="string",
            description="Path to the Python file or directory to lint.",
            required=True,
        ),
        ToolParameter(
            name="select",
            type="array",
            description="Ruff rule codes to check, e.g. ['E', 'F', 'S']. Defaults to Ruff's default selection.",
            required=False,
            default=None,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    check_types_tool,
    name="check_types",
    description=(
        "Run mypy type checker on a Python file. Returns structured type errors with "
        "error codes, line numbers, and severity. "
        "Degrades gracefully if mypy is not installed."
    ),
    parameters=[
        ToolParameter(
            name="path", type="string", description="Path to the .py file to type-check.", required=True
        ),
    ],
)

LIBRARY_REGISTRY.register(
    find_symbol_tool,
    name="find_symbol",
    description=(
        "Search all Python files under a directory for a function or class with the given name. "
        "Returns file path, line number, kind (function/class/method), and full signature per match."
    ),
    parameters=[
        ToolParameter(
            name="name",
            type="string",
            description="Python identifier to search for (function, class, or method name).",
            required=True,
        ),
        ToolParameter(
            name="directory",
            type="string",
            description="Root directory to search under.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="max_files",
            type="integer",
            description="Maximum number of files to scan.",
            required=False,
            default=200,
        ),
    ],
)

LIBRARY_REGISTRY.register(
    summarize_structure_tool,
    name="summarize_structure",
    description=(
        "Return a lightweight public-API index for all Python modules in a directory: "
        "module name, public function names, public class names per file. "
        "Use before analyze_file to identify which specific file to inspect in detail."
    ),
    parameters=[
        ToolParameter(
            name="directory",
            type="string",
            description="Root directory to index.",
            required=False,
            default=".",
        ),
        ToolParameter(
            name="max_files",
            type="integer",
            description="Maximum number of files to index.",
            required=False,
            default=200,
        ),
    ],
)


def get_library_registry() -> ToolRegistry:
    """Return the pre-populated library registry."""
    return LIBRARY_REGISTRY


def get_tool_names() -> list[str]:
    """Return names of all pre-built tools."""
    return [t.name for t in LIBRARY_REGISTRY.list_all()]


__all__ = [
    "LIBRARY_REGISTRY",
    "get_library_registry",
    "get_tool_names",
    "analyze_file_tool",
    "check_types_tool",
    "find_symbol_tool",
    "lint_code_tool",
    "summarize_structure_tool",
    "calculator",
    "file_read",
    "file_write",
    "http_request",
    "read_docx_tool",
    "read_excel_tool",
    "read_pdf_tool",
    "send_email",
    "sql_query",
    "web_search",
    "ChunkStrategy",
    "DocumentChunk",
    "DocumentMetadata",
    "ParsedDocument",
]
