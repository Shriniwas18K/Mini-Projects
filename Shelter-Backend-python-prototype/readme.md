This is python prototype for backend service which allows owners to register,login,post properties and search.It is assumed that client side session management.
Public routes : querying retrieval properties.
Private routes : querying CRUD of properties. Token recieved upon authentication needs to be given as query parameter to access the routes.

Pinecone database exists at data layer which enables retrieval of queried properties using address based on embeddings.

APIs implemented :

- /signin/

- /signup/

- /createProperty/{token} 
-- Request : JSON property object in body
-- Response : Status code 200 Property Posted successfully

- /getProperties/{address}
-- Request : address as query parameter
-- Response : Status code 200 Array of properties