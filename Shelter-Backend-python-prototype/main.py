from fastapi import FastAPI
from dotenv import load_dotenv
from datetime import datetime
from cryptography.fernet import Fernet
from pydantic import BaseModel
from typing import Literal
from pinecone import Pinecone, ServerlessSpec
from fastapi.middleware.cors import CORSMiddleware
from uuid import uuid4
import random
import psycopg2
import json
import os
from sentence_transformers import SentenceTransformer


load_dotenv()

app = FastAPI()

# Configure CORS middleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Generate a key

key = Fernet.generate_key()
cipher = Fernet(key)

DATABASE_URL=os.getenv('DATABASE_URL')
PINECONE_API_KEY=os.getenv('PINECONE_API_KEY')

'''********************************************************************
                        database connections
   ********************************************************************'''


try:
    connection = psycopg2.connect(
        DATABASE_URL
    )
    cur=connection.cursor()

except (Exception, psycopg2.Error) as error:
    print("Error while connecting to PostgreSQL:", error)


cur.execute(
    '''
    create table if not exists credentials(
        phone varchar(10),
        username varchar(20),
        password varchar(10),
        createdOn timestamp
    )
    '''
)
cur.execute(
'''
	create table if not exists transactions (
		transactionTime timestamp,
 		phone varchar(10),
  		description varchar(30)
        )
'''
)
connection.commit()

INDEX_NAME = 'pgrecommendervectordatabaseindex'
pinecone = Pinecone(api_key=PINECONE_API_KEY)

if INDEX_NAME in [index.name for index in pinecone.list_indexes()]:
  pinecone.delete_index(INDEX_NAME)

pinecone.create_index(name=INDEX_NAME, dimension=1536, metric='cosine',
  spec=ServerlessSpec(cloud='aws', region='us-east-1'))

index = pinecone.Index(INDEX_NAME)

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

'''***********************************************************
		                validation models
   ***********************************************************'''

class Auth(BaseModel):
    phone:str
    username:str
    password:str

class Property(BaseModel):
    PropertyTypes : Literal [ '1 Bedroom' , '2 Bedroom','3 Bedroom', '4 Bedroom', 'Studio']
    Security      : Literal ['Not Applicable', 'Gated Community', 'Security Guard']
    ParkingType   : Literal ['No Parking', 'Nearby Paid Parking', 'On-Street Parking','Paid Dedicated Parking', 'Free Dedicated Parking']
    LeaseTerm     : Literal ['Month to Month', '6 Months', '12 Months no extension', '12 Months with extension', 'Multi-Year']
    Background    : Literal ['Negative', 'Further Review Required', 'Neutral or Mixed', 'Standard', 'Positive']
    FurnishType   : Literal ['Fully Furnished', 'Partially Furnished', 'Unfurnished']
    RentPerPerson : int
    WifiFacility  : Literal['Available' , 'Not Available']
    Address       : str	

'''*********************************************************
  		                  Utilities  	
   *********************************************************'''

def generate_token()->str:
    '''  This function generates a token from currenttimestamp
        which is sent to client frontend, and everytime client
        has to give this token to access any of the owner routes
    '''
    generationtimestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S").encode()
    return cipher.encrypt(generationtimestamp)

def validate_token(tokenvalue)->bool:
    ''' This function checks the validity of the token ,
        one client can use one token on
        one device only for 10 min, else token will be expired 
        and session will be inactive'''
    try:
        generationtimestamp=cipher.decrypt(tokenvalue)
        generationtimestamp=datetime.strptime(
                                                generationtimestamp.decode(),
                                                "%Y-%m-%d %H:%M:%S"
                                             )
        currenttimestamp=datetime.now()
        diff=currenttimestamp-generationtimestamp
        if(diff.seconds>600):
            return False
    except:
            return False
    return True

def get_embeddings(query):
    return model.encode(query)

def upsert_to_vectordb(property):
    embeddings = get_embeddings(property.Address)
    prepped = [{'id':str(uuid4()), 'values':embeddings,
                'metadata':{'property': json.dumps(property.dict())}}]
    index.upsert(prepped)

def get_recommendations(pinecone_index, search_term, top_k=10):
    embed = get_embeddings([search_term])
    res = pinecone_index.query(vector=embed, top_k=top_k, include_metadata=True)
    return res

'''***********************************************************
	          Authentication routes
  *********************************************************'''

@app.post("/signup/")
async def sign_up(requ:Auth):
    cur.execute(f"select * from credentials where phone='{requ.phone}'")
    rows=cur.fetchall()
    if(len(rows)==1):
        return {
            "message":"user already exists try to signin"
        }
    else:
        cur.execute("insert into credentials values (%s,%s,%s,%s)",(requ.phone,requ.username,requ.password,datetime.now()))
        cur.execute("insert into transactions values(%s,%s,%s)",(datetime.now(),requ.phone,'signup'))
        connection.commit()
        return {
            "message":"user created"
        }
    
@app.post("/signin/")
async def sign_in(requ:Auth):
    '''function will check wheter username exists in database'''
    cur.execute("select * from credentials where phone=%s",(requ.phone,))
    rows=cur.fetchall()
    if(rows==[]):
        return { "message" : "user does not exists pls sign up"}
    else:
        cur.execute("insert into transactions values(%s,%s,%s)",(datetime.now(),requ.phone,'signin'))
        connection.commit()
        '''if exists then we return him token'''
        return{"token":generate_token()}


'''*************************************************************************
	          property posting and retreival routes
  ************************************************************************'''

@app.post("/createProperty/{token}/")
async def post_property(token:str,property:Property):
    '''property posted is only allowed for landlords so it requires token'''
    upsert_to_vectordb(property)
    return {"message":"Property posted successfully"}

@app.get("/getProperties/{address}/")
async def get_properties(address:str):
    reco = get_recommendations(index, address)
    for r in reco.matches:
        print(f'{r.score} : {r.metadata["property"]}')
    return {"message":"Properties recieved successfully","records":reco.matches}
