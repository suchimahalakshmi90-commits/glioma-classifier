import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()
c.execute('SELECT * FROM users')
print(c.fetchall())
conn.close()
