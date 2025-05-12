import hashlib
import os
import json
import sqlite3
from pathlib import Path
from typing import List

class ArrayHashManager:
    def __init__(self, output_dir="output", db_name="hashes.db"):
        """
        Initialize with database and file system integration.
        
        Args:
            output_dir (str): Directory for hash files
            db_name (str): Name of the SQLite database file
        """
        self.output_dir = Path(output_dir)
        self.db_path = self.output_dir / db_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self.conn = sqlite3.connect(str(self.db_path))
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        with self.conn:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS hashes (
                    hash TEXT NOT NULL,
                    file_path TEXT PRIMARY KEY,
                    last_updated REAL NOT NULL
                )
            ''')
            self.conn.execute('''
                CREATE INDEX IF NOT EXISTS idx_hash ON hashes (hash)
            ''')

    def create_hash(self, input_array: list, filename: str = None) -> str:
        """Create hash and store in both file and database"""
        # Generate hash
        array_str = json.dumps(input_array, sort_keys=True)
        hash_hex = hashlib.sha256(array_str.encode()).hexdigest()
        
        # Create filename
        filename = filename or f"array_{hash_hex[:8]}"
        file_path = self.output_dir / f"{filename}.hash.txt"
        
        # Write to file
        with open(file_path, 'w') as f:
            f.write(hash_hex)
        
        # Store in database
        self._upsert_hash_record(str(file_path), hash_hex)
        return hash_hex

    def _upsert_hash_record(self, file_path: str, hash_hex: str):
        """Update or insert hash record in database"""
        last_updated = os.path.getmtime(file_path)
        with self.conn:
            self.conn.execute('''
                INSERT OR REPLACE INTO hashes 
                (hash, file_path, last_updated)
                VALUES (?, ?, ?)
            ''', (hash_hex, file_path, last_updated))

    def sync_database(self, search_root: str = "/"):
        """Synchronize database with file system"""
        for root, _, files in os.walk(search_root):
            for file in files:
                if file.endswith('.hash.txt'):
                    file_path = Path(root) / file
                    self._process_file(file_path)

    def _process_file(self, file_path: Path):
        """Process a single file for synchronization"""
        try:
            # Get current file stats
            last_updated = file_path.stat().st_mtime
            current_hash = file_path.read_text().strip()
            
            # Check database record
            db_record = self.conn.execute('''
                SELECT hash, last_updated FROM hashes
                WHERE file_path = ?
            ''', (str(file_path),)).fetchone()
            
            if not db_record or db_record[1] != last_updated or db_record[0] != current_hash:
                self._upsert_hash_record(str(file_path), current_hash)
        except (OSError, PermissionError) as e:
            print(f"Skipping {file_path}: {str(e)}")

    def find_matches(self, target_hash: str, search_root: str = None) -> List[str]:
        """
        Find matches using database with optional filesystem verification
        
        Args:
            target_hash: Hash value to search for
            search_root: Optional root directory to limit search
        """
        # Database query
        query = 'SELECT file_path FROM hashes WHERE hash = ?'
        params = [target_hash]
        
        if search_root:
            query += ' AND file_path LIKE ?'
            params.append(f"{search_root}%")
        
        matches = [row[0] for row in self.conn.execute(query, params)]
        
        # Optional verification
        verified_matches = []
        for path in matches:
            try:
                if Path(path).read_text().strip() == target_hash:
                    verified_matches.append(path)
            except (OSError, PermissionError):
                continue
                
        return verified_matches

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()