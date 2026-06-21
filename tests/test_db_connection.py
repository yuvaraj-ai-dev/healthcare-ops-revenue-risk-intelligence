import os
import pymysql
from dotenv import load_dotenv

# Load env variables
load_dotenv()

host = os.getenv("MYSQL_HOST", "localhost")
port = int(os.getenv("MYSQL_PORT", 3306))
user = os.getenv("MYSQL_USER", "root")
password = os.getenv("MYSQL_PASSWORD", "")
db = os.getenv("MYSQL_DATABASE", "hospital_db")

print(f"Attempting to connect to MySQL at {host}:{port} as user '{user}' to database '{db}'...")

try:
    # Connect to MySQL server (without specifying DB first, to verify server access)
    conn = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password
    )
    print("Successfully connected to MySQL Server!")
    
    # Check if database exists, if not create it
    with conn.cursor() as cursor:
        cursor.execute("SHOW DATABASES")
        databases = [row[0] for row in cursor.fetchall()]
        print("Available databases:", databases)
        
        if db not in databases:
            print(f"Database '{db}' does not exist. Attempting to create it...")
            cursor.execute(f"CREATE DATABASE {db}")
            print(f"Database '{db}' created successfully!")
        else:
            print(f"Database '{db}' exists.")
            
    conn.close()
    
    # Try connecting to the specific database
    conn_db = pymysql.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        database=db
    )
    print(f"Successfully connected to database '{db}'!")
    conn_db.close()
    print("Database connection test PASSED!")
    
except Exception as e:
    print("Database connection test FAILED!")
    print("Error:", str(e))
