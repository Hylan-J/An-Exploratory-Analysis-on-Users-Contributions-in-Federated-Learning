import sqlite3


class SQLiteHelper:
    def __init__(self):
        self.database_connect = None
        self.database_cursor = None

    def create_database(self, database_path):
        self.database_connect = sqlite3.connect(database_path)
        # 获得数据库游标
        self.database_cursor = self.database_connect.cursor()

    def create_experiment_table(self):
        # --------------------------------------------------------------------------------------------------------------
        # |                                                实验参数表                                                    |
        # |------------------------------------------------------------------------------------------------------------|
        # |     id      | indicator_type |   dataset   |   client_data_number   |   noise_ratio   |  malicious_client  |
        # |数据表中的条目id|     指标类型    |    数据集    |       客户端数据量       |     噪声比例      |      恶意客户端      |
        # --------------------------------------------------------------------------------------------------------------
        # |  learning_rate  |   batch_size    |    momentum   |
        # |       学习率     |    批处理大小     |      动量      |
        # --------------------------------------------------------------------------------------------------------------
        # 创建数据库中的实验参数表
        create_SQL = ("CREATE TABLE IF NOT EXISTS experiment(id integer PRIMARY KEY, "
                      "indicator_type text NOT NULL, "
                      "dataset text NOT NULL, "
                      "client_data_number text NOT NULL, "
                      "noise_ratio text NOT NULL,"
                      "malicious_client text NOT NULL, "
                      "learning_rate text NOT NULL, "
                      "batch_size INTEGER NOT NULL, "
                      "momentum text NOT NULL) ")
        self.database_cursor.execute(create_SQL)
        self.database_connect.commit()

    def insert_experiment_table(self,
                                indicator_type: str,
                                dataset: str,
                                client_data_number: str,
                                noise_ratio: str,
                                malicious_client: str,
                                learning_rate: str,
                                batch_size: int,
                                momentum: str):
        experiment_content = [indicator_type,
                              dataset,
                              client_data_number,
                              noise_ratio,
                              malicious_client,
                              learning_rate,
                              batch_size,
                              momentum]
        insert_SQL = ("INSERT INTO experiment(indicator_type,dataset,client_data_number,noise_ratio,"
                      "malicious_client,learning_rate,batch_size,momentum) VALUES(?,?,?,?,?,?,?,?)")
        self.database_cursor.execute(insert_SQL, experiment_content)
        self.database_connect.commit()

    def create_clients_table(self):
        # -----------------------------------------------------------------------------------------
        # |                                  客户端学习情况表                                        |
        # |---------------------------------------------------------------------------------------|
        # |      id       |    epoch   | client_id |       loss         |         accuracy        |
        # | 数据表中的条目id | 训练对应的轮次| 客户端的id | 客户端在epoch时的loss | 客户端在epoch时的accuracy |
        # -----------------------------------------------------------------------------------------
        # 创建数据库中的客户端学习情况表
        create_SQL = ("CREATE TABLE IF NOT EXISTS clients(id integer PRIMARY KEY, "
                      "epoch INTEGER NOT NULL, "
                      "client_id INTEGER NOT NULL, "
                      "loss text NOT NULL, "
                      "accuracy text NOT NULL) ")
        self.database_cursor.execute(create_SQL)
        self.database_connect.commit()

    def insert_clients_table(self,
                             epoch: int,
                             client_id: int,
                             loss: str,
                             accuracy: str):
        clients_content = [epoch,
                           client_id,
                           loss,
                           accuracy]
        insert_SQL = "INSERT INTO clients(epoch,client_id,loss,accuracy) VALUES(?,?,?,?)"
        self.database_cursor.execute(insert_SQL, clients_content)
        self.database_connect.commit()

    def create_server_table(self):
        # --------------------------------------------------
        # |                  服务器影响情况表                 |
        # |------------------------------------------------|
        # |       id       |    chosen   | indicator_value |
        # | 数据表中的条目id  | 被选中的客户端 |     指标数值     |
        # --------------------------------------------------
        # 创建数据库中的服务器影响情况表
        create_SQL = ("CREATE TABLE IF NOT EXISTS server(id integer PRIMARY KEY, "
                      "chosen text NOT NULL, "
                      "indicator_value text NOT NULL) ")
        self.database_cursor.execute(create_SQL)
        self.database_connect.commit()

    def insert_server_table(self,
                            chosen: str,
                            indicator_value: str):
        server_content = [chosen, indicator_value]
        insert_SQL = "INSERT INTO server(chosen,indicator_value) VALUES(?,?)"
        self.database_cursor.execute(insert_SQL, server_content)
        self.database_connect.commit()

    def create_tables(self):
        self.create_experiment_table()
        self.create_clients_table()
        self.create_server_table()

    def close(self):
        self.database_cursor.close()
        self.database_connect.close()
