"""
ingest.py - Load course documents into ChromaDB vector store.

Usage:
    python ingest.py

Place your course PDFs or .txt files in the ./docs/ folder before running.
This script chunks them and stores embeddings in a local ChromaDB database.
"""

import os
import glob
import chromadb
from chromadb.utils import embedding_functions
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader


DOCS_DIR = os.path.join(os.path.dirname(__file__), "docs")
CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text


def extract_text_from_txt(txt_path: str) -> str:
    """Read a plain text file."""
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def load_documents() -> list[dict]:
    """Load all documents from the docs directory."""
    documents = []

    # Load PDFs
    for pdf_path in glob.glob(os.path.join(DOCS_DIR, "*.pdf")):
        print(f"  Loading PDF: {os.path.basename(pdf_path)}")
        text = extract_text_from_pdf(pdf_path)
        if text.strip():
            documents.append({
                "text": text,
                "source": os.path.basename(pdf_path)
            })

    # Load text files
    for txt_path in glob.glob(os.path.join(DOCS_DIR, "*.txt")):
        print(f"  Loading TXT: {os.path.basename(txt_path)}")
        text = extract_text_from_txt(txt_path)
        if text.strip():
            documents.append({
                "text": text,
                "source": os.path.basename(txt_path)
            })

    return documents


def chunk_documents(documents: list[dict], chunk_size=500, chunk_overlap=50) -> list[dict]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "id": f"{doc['source']}_chunk_{i}",
                "text": chunk_text,
                "source": doc["source"],
                "chunk_index": i
            })

    return chunks


def ingest():
    """Main ingestion pipeline: load docs -> chunk -> embed -> store in ChromaDB."""
    print("=" * 60)
    print("Course Document Ingestion Pipeline")
    print("=" * 60)

    # Check if docs directory has files
    if not os.path.exists(DOCS_DIR):
        os.makedirs(DOCS_DIR)

    all_files = glob.glob(os.path.join(DOCS_DIR, "*.pdf")) + \
                glob.glob(os.path.join(DOCS_DIR, "*.txt"))

    if not all_files:
        print(f"\nNo documents found in '{DOCS_DIR}/'")
        print("Please add .pdf or .txt files to the docs/ folder.")
        print("\nCreating a sample document for demo purposes...")
        create_sample_docs()

    # Load documents
    print("\n[1/3] Loading documents...")
    documents = load_documents()
    print(f"  Loaded {len(documents)} document(s)")

    # Chunk documents
    print("\n[2/3] Chunking documents...")
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks")

    # Store in ChromaDB
    print("\n[3/3] Storing in ChromaDB...")
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Delete existing collection if it exists, then recreate
    try:
        client.delete_collection("course_docs")
    except Exception:
        pass

    collection = client.get_or_create_collection(
        name="course_docs",
        embedding_function=ef,
        metadata={"description": "Course documents for Q&A"}
    )

    # Add chunks in batches
    batch_size = 50
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[{"source": c["source"], "chunk_index": c["chunk_index"]} for c in batch]
        )

    print(f"  Stored {len(chunks)} chunks in ChromaDB at '{CHROMA_DIR}'")
    print(f"\n{'=' * 60}")
    print("Ingestion complete! You can now run: python main.py")
    print(f"{'=' * 60}")


def create_sample_docs():
    """Create sample course documents for demonstration."""

    sample_1 = """Data Warehousing Fundamentals

A data warehouse is a centralized repository designed to store integrated data from multiple sources
for the purpose of reporting and data analysis. It supports business intelligence (BI) activities.

Star Schema Design:
The star schema is the simplest form of dimensional modeling. It consists of a central fact table
surrounded by dimension tables. The fact table contains measurable, quantitative data (facts) such
as sales amount, quantity sold, and profit. Dimension tables contain descriptive attributes related
to the facts, such as product name, customer demographics, and time periods.

Key characteristics of a star schema:
- One central fact table connected to multiple dimension tables
- Dimension tables are denormalized (no sub-dimensions)
- Simple queries with fewer joins compared to snowflake schema
- Optimized for read-heavy analytical workloads

Snowflake Schema:
The snowflake schema is a normalized version of the star schema. Dimension tables are broken
into sub-dimension tables, reducing data redundancy but increasing query complexity due to
additional joins.

Inmon vs Kimball Approach:
Bill Inmon's approach (top-down) advocates building an enterprise data warehouse first, then
creating data marts. Ralph Kimball's approach (bottom-up) starts with individual data marts
that are later integrated. Inmon emphasizes normalization and a single source of truth,
while Kimball focuses on dimensional modeling and faster time-to-value.

ETL Process:
ETL stands for Extract, Transform, Load. It is the process of:
1. Extract: Pulling data from source systems (databases, APIs, files)
2. Transform: Cleaning, validating, and restructuring data
3. Load: Writing the processed data into the data warehouse

OLAP (Online Analytical Processing):
OLAP enables complex analytical queries on multidimensional data. Key operations include:
- Slice: Selecting a single dimension value
- Dice: Selecting multiple dimension values
- Drill-down: Moving to a more detailed level
- Roll-up: Aggregating data to a higher level
- Pivot: Rotating the data axes for different perspectives
"""

    sample_2 = """Computer Security Fundamentals

Confidentiality, Integrity, and Availability (CIA Triad):
The CIA triad is the foundational model for information security:
- Confidentiality: Ensuring data is accessible only to authorized parties
- Integrity: Ensuring data has not been tampered with or altered
- Availability: Ensuring systems and data are accessible when needed

Symmetric vs Asymmetric Encryption:
Symmetric encryption uses the same key for both encryption and decryption (e.g., AES, DES).
It is fast but requires secure key distribution. Asymmetric encryption uses a public-private
key pair (e.g., RSA, ECC). The public key encrypts, and only the corresponding private key
can decrypt. It solves the key distribution problem but is slower.

Hash Functions:
A cryptographic hash function takes an input and produces a fixed-size output (digest).
Properties: deterministic, fast to compute, pre-image resistant, collision resistant.
Common algorithms: SHA-256, SHA-3, MD5 (deprecated due to collisions).

Digital Signatures:
A digital signature provides authentication, integrity, and non-repudiation.
Process: The sender hashes the message, then encrypts the hash with their private key.
The receiver decrypts using the sender's public key and compares hashes.

OpenSSL:
OpenSSL is a widely-used open-source toolkit for TLS/SSL protocols and cryptographic operations.
Common commands:
- openssl enc -aes-256-cbc -in file.txt -out encrypted.txt  (encrypt a file)
- openssl dgst -sha256 file.txt  (compute SHA-256 hash)
- openssl genrsa -out private.pem 2048  (generate RSA private key)
- openssl rsa -in private.pem -pubout -out public.pem  (extract public key)

Network Security:
Firewalls filter network traffic based on predefined rules. Types include packet filtering,
stateful inspection, and application-layer firewalls. IDS (Intrusion Detection Systems)
monitor traffic for suspicious activity, while IPS (Intrusion Prevention Systems) can
actively block threats.

Access Control Models:
- DAC (Discretionary Access Control): Owner decides permissions
- MAC (Mandatory Access Control): System enforces labels/clearances
- RBAC (Role-Based Access Control): Permissions assigned to roles, users assigned to roles
"""

    sample_3 = """5G Networking and Telecommunications

Evolution of Mobile Networks:
1G (analog voice) -> 2G (digital voice, SMS) -> 3G (mobile data, video calls) ->
4G LTE (high-speed broadband) -> 5G NR (ultra-low latency, massive IoT)

5G Architecture:
5G uses a Service-Based Architecture (SBA) where network functions communicate via APIs.
Key components:
- gNodeB (gNB): The 5G base station that handles radio communication
- AMF (Access and Mobility Management Function): Manages UE registration and mobility
- SMF (Session Management Function): Manages PDU sessions
- UPF (User Plane Function): Handles data packet routing and forwarding
- NSSF (Network Slice Selection Function): Selects appropriate network slice

Network Slicing:
Network slicing allows operators to create multiple virtual networks on shared physical
infrastructure. Each slice is optimized for specific use cases:
- eMBB (enhanced Mobile Broadband): High data rates for streaming/downloads
- URLLC (Ultra-Reliable Low-Latency Communication): For autonomous vehicles, remote surgery
- mMTC (massive Machine-Type Communication): For IoT sensors, smart cities

OpenAirInterface (OAI):
OAI is an open-source software implementation of 5G. It includes:
- OAI 5G Core: Implements AMF, SMF, UPF, and other core network functions in Docker
- OAI gNodeB: Software-defined 5G base station
- OAI UE: Software-defined 5G user equipment

USRP (Universal Software Radio Peripheral):
USRP devices like the B210 are software-defined radios used to transmit/receive RF signals.
Combined with OAI, they enable building real standalone 5G networks for research.

Key 5G parameters:
- Frequency bands: Sub-6 GHz (wider coverage) and mmWave (higher speeds)
- PLMN (Public Land Mobile Network): Identified by MCC + MNC
- TAC (Tracking Area Code): Identifies tracking areas for UE mobility management
"""

    with open(os.path.join(DOCS_DIR, "data_warehousing.txt"), "w") as f:
        f.write(sample_1)

    with open(os.path.join(DOCS_DIR, "computer_security.txt"), "w") as f:
        f.write(sample_2)

    with open(os.path.join(DOCS_DIR, "5g_networking.txt"), "w") as f:
        f.write(sample_3)

    print("  Created 3 sample course documents in docs/")


if __name__ == "__main__":
    ingest()