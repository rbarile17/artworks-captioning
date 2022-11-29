from neo4j import GraphDatabase
import pandas as pd
from . import ARTGRAPH_PATH

class Neo4jConnection:
    def __init__(self, uri, user, pwd):
        self.__uri = uri
        self.__user = user
        self.__pwd = pwd
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
    
def main():
    conn = Neo4jConnection(uri="bolt://localhost:7687", user="neo4j", pwd="artgraph")
    artworks = conn.query('MATCH (n:Artwork) RETURN n', db='neo4j')
    artworks = [artwork['n'] for artwork in artworks]

    artworks_df = pd.DataFrame(artworks)

    # rename name column to file_name
    artworks_df = artworks_df.rename(columns={'name': 'file_name'})
    artworks_df.to_csv(ARTGRAPH_PATH / 'artgraph.csv', index=False)

if __name__ == "__main__":
    main()