import sqlite3 as db
import os


class get_content:
    '''
    input: 数据库id 数据库中表名
    output: {}以表名为key 以该表内容为value的字典
    '''
    def __init__(self,db_path = '../data/database/'):
        self.db_path = db_path

    def get_content(self, db_id = 'bike_1', tables =["station","status","trip","weather"]):
        path = self.db_path + db_id + '/' + db_id + '.sqlite'
        conn = db.connect(path)
        conn.text_factory = str
        cursor = conn.cursor()

        values = {}

        for table in tables:
            try:
                cursor.execute(f"SELECT * FROM {table} LIMIT 5000")
                values[table] = cursor.fetchall()
            except:
                conn.text_factory = lambda x: str(x, 'latin1')
                cursor = conn.cursor()
                cursor.execute(f"SELECT * FROM {table} LIMIT 5000")
                values[table] = cursor.fetchall()

        return values




print(get_content().get_content().keys())
print(get_content().get_content()['station'])