from typing import Union
from fastapi import FastAPI

# Import Union since our Item object will have tags that can be strings or a list.
from typing import Union

# BaseModel from Pydantic is used to define data objects.
from pydantic import BaseModel

# Declare the data object with its components and their type.
class TaggedItem(BaseModel):
    name:str
    tags: Union[str,list]
    item_id:int
    
class Value(BaseModel):
    value: int

app =FastAPI()

# This allows sending of data (our TaggedItem) via POST to the API.
@app.post("/items/")
async def create_item(item: TaggedItem):
   return item

@app.get("/")
def read_root():
   return {"Hello":"World"}

@app.get("/items/{item_id}")
def read_item(item_id:int,q:Union[str,None] =None):
   return {"item_id":item_id,"q":q}

# Use POST action to send data to the server
@app.post("/{path}")
async def exercise_function(path:int,query:int,body: Value):
   return {"path":path,"query":query,"body":body}
