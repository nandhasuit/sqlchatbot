from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from database import SQLDatabase
from langchain_mistralai.chat_models import ChatMistralAI 
import os
from dotenv import load_dotenv
load_dotenv()
def get_sql_chain(db):
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Understand the Schema, Tables (artists,albums,tracks,media_types,genres,playlists,playlist_track,customers,employees,invoices,invoice_items)and its column definition details below, write a SQL query that would answer the user's question. Take the conversation history into account.
    

    Note:  Write only the SQL query for MYsql database, 
    Note:  Do not wrap the SQL query in any other text, not even backticks.

    Database context:

    The Chinook data model represents a digital media store, including tables for artists, albums, media tracks, invoices, and customers.
    Media-related data was created using real data from an Apple iTunes library.
    Customer and employee information was created using fictitious names and addresses that can be located on Google maps, and other well formatted data (phone, fax, email, etc.)
    Sales information was auto generated using random data for a four year period.
    The Chinook sample database includes: 11 tables
    A variety of indexes, primary and foreign key constraints
    Over 15,000 rows of data
    
    Schema: <SCHEMA>{schema}</SCHEMA>

    Tables:
    artist: Stores artist data, including the artist ID and name.
    album: Stores album data, including the album ID, title, and the associated artist ID.
    track: Stores track data, including the track ID, name, album ID, media type, genre, composer, milliseconds, bytes, and unit price.
    media_type: Stores the different media types, such as MPEG audio and AAC audio files.
    genre: Stores the different music genre, such as rock, jazz, and metal.
    playlist: Stores playlist data, including the playlist ID and name.
    playlist_track: A junction table that links the playlists and tracks tables, representing the many-to-many relationship between playlists and tracks.
    customer: Stores customer data, including the customer ID, first name, last name, company, address, city, state, country, postal code, phone, fax, and email.
    employee: Stores employee data, including the employee ID, last name, first name, title, reports to, birth date, hire date, address, city, state, country, postal code, phone, and fax.
    invoice: Stores invoice data, including the invoice ID, customer ID, invoice date, billing address, billing city, billing state, billing country, billing postal code, total, and the invoice status.
    invoiceline: Stores the individual line items for each invoice, including the invoice ID, track ID, unit price, and quantity.

    Column Definitions:

    Based on the information provided in the search results, here are some key column definitions for the Chinook database:
      artist:
        ArtistId: Unique identifier for the artist
        Name: Name of the artist
      album:
        AlbumId: Unique identifier for the album
        Title: Title of the album
        ArtistId: Foreign key linking the album to the artist
      track:
        TrackId: Unique identifier for the track
        Name: Title of the track
        AlbumId: Foreign key linking the track to the album
        MediaTypeId: Foreign key linking the track to the media type
        GenreId: Foreign key linking the track to the genre
        Composer: Name of the composer
        Milliseconds: Duration of the track in milliseconds
        Bytes: Size of the track in bytes
        UnitPrice: Price of the track
      customer:
        CustomerId: Unique identifier for the customer
        FirstName: First name of the customer
        LastName: Last name of the customer
        Company: Company name of the customer
        Address: Address of the customer
        City: City of the customer
        State: State of the customer
        Country: Country of the customer
        PostalCode: Postal code of the customer
        Phone: Phone number of the customer
        Fax: Fax number of the customer
        Email: Email address of the customer
        SupportRepId:a
      invoice:
        InvoiceId: Unique identifier for the invoice
        CustomerId: Foreign key linking the invoice to the customer
        InvoiceDate: Date of the invoice
        BillingAddress: Billing address of the invoice
        BillingCity: Billing city of the invoice
        BillingState: Billing state of the invoice
        BillingCountry: Billing country of the invoice
        BillingPostalCode: Billing postal code of the invoice
        Total: Total amount of the invoice
  This should provide a good overview of the Chinook database schema and the key column definitions to help you work with the data.
   
    
    For example below is the template where we ask questiona and you should generate similar to the below examples :
      Question: List all unique genres
      SQL Query: SELECT DISTINCT GenreId, Name FROM Genre;

      Question: List all customers and their respective countries:
      SQL Query: SELECT CustomerId, FirstName, LastName, Country 
                  FROM Customer;
      Question: List all tracks with their corresponding album titles:
      SQL Query: SELECT t.TrackId, t.Name AS TrackName, a.Title AS AlbumTitle
                  FROM Track t
                  INNER JOIN Album a ON t.AlbumId = a.AlbumId;
      Question: List top 10 tracks with the highest unit price:
      SQL Query: SELECT TrackId, Name, UnitPrice
                  FROM Track
                  ORDER BY UnitPrice DESC
                  LIMIT 10;
      Question: List all invoices along with their total amounts and customer names:
      SQL Query: SELECT i.InvoiceId, i.Total, c.FirstName, c.LastName
                  FROM Invoice i
                  INNER JOIN Customer c ON i.CustomerId = c.CustomerId; 
      Question: List all artists and the number of tracks they have in the database:
      SQL Query: SELECT ar.ArtistId, ar.Name AS ArtistName, COUNT(t.TrackId) AS TrackCount
                  FROM Artist ar
                  INNER JOIN Album al ON ar.ArtistId = al.ArtistId
                  INNER JOIN Track t ON al.AlbumId = t.AlbumId
                  GROUP BY ar.ArtistId, ar.Name
                  ORDER BY TrackCount DESC;
    
    Note: Do not wrap the SQL query in any other text, not even backticks.

    Your turn:
    
    Question: {question}
    SQL Query:

        """
    
    prompt = ChatPromptTemplate.from_template(template)
  
    llm = ChatGroq(temperature=0,model="mixtral-8x7b-32768",api_key=os.environ.get("GROQ_API_KEY"))    
    def get_schema(_):
        return db.get_table_info()
  
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
    sql_chain = get_sql_chain(db)
  
    template = """
        You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
        Based on the table schema below, question, sql query, and sql response, write a natural language response.
        <SCHEMA>{schema}</SCHEMA>

        SQL Query: <SQL>{query}</SQL>
        User question: {question}
        SQL Response: {response}"""
  
    prompt = ChatPromptTemplate.from_template(template)
  
    llm = ChatGroq(temperature=0,model="mixtral-8x7b-32768",api_key=os.environ.get("GROQ_API_KEY"))
    
    chain = (
        RunnablePassthrough.assign(query=sql_chain).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
  
    return chain.invoke({
        "question": user_query,
        "chat_history": chat_history,
    })
