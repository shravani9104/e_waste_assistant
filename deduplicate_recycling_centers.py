import sqlite3
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

DB_PATH = 'ewaste_assistant.db'

def normalize_address(address):
    return address.strip().lower()

def deduplicate_recycling_centers():
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Retrieve all recycling centers ordered by id (to keep first entry)
        cursor.execute('''
            SELECT id, address FROM recycling_centers ORDER BY id
        ''')
        rows = cursor.fetchall()

        seen_addresses = set()
        duplicates = []

        for row in rows:
            id_, address = row
            norm_addr = normalize_address(address)
            if norm_addr in seen_addresses:
                duplicates.append(id_)
            else:
                seen_addresses.add(norm_addr)

        if duplicates:
            logger.info(f"Deleting {len(duplicates)} duplicate recycling center entries...")
            cursor.executemany('DELETE FROM recycling_centers WHERE id = ?', [(id_,) for id_ in duplicates])
            conn.commit()
            logger.info("Duplicates removed successfully.")
        else:
            logger.info("No duplicates found.")

        conn.close()
    except sqlite3.Error as e:
        logger.error(f"Error during deduplication: {e}")

if __name__ == '__main__':
    deduplicate_recycling_centers()
