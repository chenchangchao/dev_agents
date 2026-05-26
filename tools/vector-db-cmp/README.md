## pgvector and Qdrant
Choosing Between pgvector and Qdrant for Large-Scale Vector Database on Azure – What Do You Recommend?

I worked extensively with both.
pgVector:
Pros: Amazing productivity, easily maintainable, and you get all the stability and upside of Postgres.
Cons: Performance tend to be slightly behind Qdrant. At certain scale, becomes hard to work with.

Qdrant:
Pros: Best in class performance, flexible scaling.
Cons: You will need 2 DBs, and this is a lot of engineering hours to get right.

Overall, I like pgVector and I have a large open-source RAG API built on it. But, Qdrant is really good, and it feels like the pgVector community is always catching up to something Qdrant have or does.