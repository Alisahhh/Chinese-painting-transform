import sqlite3
import time
import os

conn = sqlite3.connect('test.db')
c = conn.cursor()
print "Opened database successfully"
c = conn.cursor()
c.execute('''CREATE TABLE Painting
       (ID INT PRIMARY KEY     NOT NULL,
       STATUS INT      NOT NULL,
       RECVPATH        CHAR(50),
       LOADPATH        CHAR(50));''')
print "Table created successfully"
conn.commit()
conn.close()