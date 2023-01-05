from neo4j import GraphDatabase
import pandas as pd
from .. import ARTGRAPH_PATH
from ... import load_params

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
            session = self.__driver.session(database=db) \
                if db is not None else self.__driver.session() 
            response = list(session.run(query))
        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return response
    
def main(params=None):
    if params is None:
        params = load_params()["retrieve_artgraph_artworks"]

    conn = Neo4jConnection(
        uri=params["neo4j_uri"], 
        user=params["neo4j_user"], 
        pwd=params["neo4j_pwd"]
    )

    artworks = conn.query('MATCH (n:Artwork) RETURN n', db=params["neo4j_db"])
    artworks = [artwork['n'] for artwork in artworks]

    artworks_df = pd.DataFrame(artworks)

    # rename name column to file_name
    artworks_df = artworks_df.rename(columns={'name': 'file_name'})
    artworks_df.to_csv(ARTGRAPH_PATH / 'artgraph.csv', index=False)

if __name__ == "__main__":
    main()