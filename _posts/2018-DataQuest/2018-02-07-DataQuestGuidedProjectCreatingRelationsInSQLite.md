---
title: Dataquest Guided Project - Creating Relations In SQLite
date: 2018-02-07 00:00 +/-0000
categories: [DataQuest]
tags: [dataquest, basics]
image:
  path: /posts_images/2018-02-07-DataQuestGuidedProjectCreatingRelationsInSQLite/cover.PNG
---

In this project we will use the cleaned data from the previous mission and work mainly in SQL. We will use SQL to transform the nominations tables into new tables, and create relations across these tables.

First let's take a look at the table's schema and the first ten rows.


```python
import sqlite3
conn = sqlite3.connect("nominations.db")
cursor = conn.cursor()
```


```python
q1 = '''
PRAGMA table_info(nominations)
'''

schema = cursor.execute(q1).fetchall()
```


```python
q2 = '''
SELECT * FROM nominations LIMIT 10
'''

first_ten = cursor.execute(q2).fetchall()
first_ten

for item in first_ten:
    print(item)
```

    (2010, 'Actor -- Leading Role', 'Javier Bardem', 0, 'Biutiful', 'Uxbal')
    (2010, 'Actor -- Leading Role', 'Jeff Bridges', 0, 'True Grit', 'Rooster Cogburn')
    (2010, 'Actor -- Leading Role', 'Jesse Eisenberg', 0, 'The Social Network', 'Mark Zuckerberg')
    (2010, 'Actor -- Leading Role', 'Colin Firth', 1, "The King's Speech", 'King George VI')
    (2010, 'Actor -- Leading Role', 'James Franco', 0, '127 Hours', 'Aron Ralston')
    (2010, 'Actor -- Supporting Role', 'Christian Bale', 1, 'The Fighter', 'Dicky Eklund')
    (2010, 'Actor -- Supporting Role', 'John Hawkes', 0, "Winter's Bone", 'Teardrop')
    (2010, 'Actor -- Supporting Role', 'Jeremy Renner', 0, 'The Town', 'James Coughlin')
    (2010, 'Actor -- Supporting Role', 'Mark Ruffalo', 0, 'The Kids Are All Right', 'Paul')
    (2010, 'Actor -- Supporting Role', 'Geoffrey Rush', 0, "The King's Speech", 'Lionel Logue')
    


```python
for item in schema:
    print(item)
```

    (0, 'Year', 'INTEGER', 0, None, 0)
    (1, 'Category', 'TEXT', 0, None, 0)
    (2, 'Nominee', 'TEXT', 0, None, 0)
    (3, 'Won', 'INTEGER', 0, None, 0)
    (4, 'Movie', 'TEXT', 0, None, 0)
    (5, 'Character', 'TEXT', 0, None, 0)
    

We want to create a new table within the databse named "ceremonies". This table will have the Year and the Host only


```python
q3 = '''
    CREATE TABLE ceremonies (
    id INTEGER primary key,
    Year INTEGER,
    Host TEXT
);
'''

years_hosts = [(2010, "Steve Martin"),
               (2009, "Hugh Jackman"),
               (2008, "Jon Stewart"),
               (2007, "Ellen DeGeneres"),
               (2006, "Jon Stewart"),
               (2005, "Chris Rock"),
               (2004, "Billy Crystal"),
               (2003, "Steve Martin"),
               (2002, "Whoopi Goldberg"),
               (2001, "Steve Martin"),
               (2000, "Billy Crystal"),
            ]
insert_query = "INSERT INTO ceremonies (Year, Host) VALUES (?, ?)"

```

The .executemany() method allows us to perform many insert commands at once, until the list ends


```python
conn.execute(q3)

conn.executemany(insert_query, years_hosts)
```




    <sqlite3.Cursor at 0x1cefc369030>



Let's check out the table we just created .


```python
q4 = "SELECT * FROM ceremonies LIMIT 10"
q5 = "PRAGMA table_info(ceremonies)"

result = cursor.execute(q4)
result = result.fetchall()
result
```




    [(1, 2010, 'Steve Martin'),
     (2, 2009, 'Hugh Jackman'),
     (3, 2008, 'Jon Stewart'),
     (4, 2007, 'Ellen DeGeneres'),
     (5, 2006, 'Jon Stewart'),
     (6, 2005, 'Chris Rock'),
     (7, 2004, 'Billy Crystal'),
     (8, 2003, 'Steve Martin'),
     (9, 2002, 'Whoopi Goldberg'),
     (10, 2001, 'Steve Martin')]




```python
result = cursor.execute(q5)
result = result.fetchall()
result
```




    [(0, 'id', 'INTEGER', 0, None, 1),
     (1, 'Year', 'INTEGER', 0, None, 0),
     (2, 'Host', 'TEXT', 0, None, 0)]



We want to avoid editing the original table. So let's make a new nominations table, but this time we'll include the ceremony_id as a foriegn key. 


```python
q6 = "PRAGMA foreign_keys = ON;"
cursor.execute(q6)
```




    <sqlite3.Cursor at 0x1cefb338ea0>




```python
q7 = '''
    CREATE TABLE nominations_two (
    id integer primary key,
    category text,
    nominee text,
    movie text,
    character text,
    won integer,
    ceremony_id integer,
    foreign key(ceremony_id) references ceremonies(id)
);
'''

#Query to be inserted into the new table
insert_query2 = "INSERT INTO nominations_two (category, nominee, movie, character, won, ceremony_id) VALUES (?, ?, ?, ?, ?, ?)"

```


```python
cursor.execute(q7)
```




    <sqlite3.Cursor at 0x1cefb338ea0>



We'll need a list of tuples, we can do this by writing a query and setting the results to a list.


```python
q8 = '''
SELECT nominations.category, nominations.nominee, nominations.movie, nominations.character, nominations.won, ceremonies.id
FROM nominations
INNER JOIN ceremonies
ON nominations.year == ceremonies.year
'''
#returns a list of tuples
joined_nominations = cursor.execute(q8).fetchall()
joined_nominations[0:5]
```




    [('Actor -- Leading Role', 'Javier Bardem', 'Biutiful', 'Uxbal', 0, 1),
     ('Actor -- Leading Role',
      'Jeff Bridges',
      'True Grit',
      'Rooster Cogburn',
      0,
      1),
     ('Actor -- Leading Role',
      'Jesse Eisenberg',
      'The Social Network',
      'Mark Zuckerberg',
      0,
      1),
     ('Actor -- Leading Role',
      'Colin Firth',
      "The King's Speech",
      'King George VI',
      1,
      1),
     ('Actor -- Leading Role', 'James Franco', '127 Hours', 'Aron Ralston', 0, 1)]




```python
#populating the nominations_two table
conn.executemany(insert_query2, joined_nominations)
```




    <sqlite3.Cursor at 0x1cefc369180>



 Let's check out the new table, "nominations_two" that we just made.


```python
q9 = "SELECT * FROM nominations_two LIMIT 5"
result = cursor.execute(q9)
result = result.fetchall()
result[0:5]
```




    [(1, 'Actor -- Leading Role', 'Javier Bardem', 'Biutiful', 'Uxbal', 0, 1),
     (2,
      'Actor -- Leading Role',
      'Jeff Bridges',
      'True Grit',
      'Rooster Cogburn',
      0,
      1),
     (3,
      'Actor -- Leading Role',
      'Jesse Eisenberg',
      'The Social Network',
      'Mark Zuckerberg',
      0,
      1),
     (4,
      'Actor -- Leading Role',
      'Colin Firth',
      "The King's Speech",
      'King George VI',
      1,
      1),
     (5,
      'Actor -- Leading Role',
      'James Franco',
      '127 Hours',
      'Aron Ralston',
      0,
      1)]



We can change the table name of nominations_two back to nominations, replacing the table.


```python
#We can drop the table "nominations" with the DROP TABLE sql command
q10 = "DROP TABLE nominations"
conn.execute(q10)

q11 = "ALTER TABLE nominations_two RENAME TO nominations"
conn.execute(q11)

```




    <sqlite3.Cursor at 0x1cefc369260>



We can repeat the process and create three new tables and use have their IDs cross reference each other.


```python
q11 = '''
    CREATE TABLE movies(
    id integer PRIMARY KEY,
    movie text
)'''


q12 = '''
    CREATE TABLE actors(
    id integer PRIMARY KEY,
    actor text
)'''

q13 = '''
    CREATE TABLE movies_actors(
    id integer PRIMARY KEY,
    movie_id integer references movies(id),
    actor_id integer references actors(id)
    );
'''
```


```python
conn.execute(q11)
conn.execute(q12)

```




    <sqlite3.Cursor at 0x1cefc369340>




```python
conn.execute(q13)
```




    <sqlite3.Cursor at 0x1cefc3692d0>




```python
insert_into_movies = "INSERT INTO movies (movie) SELECT nominations.movie from nominations"
insert_into_actors = "INSERT INTO actors (actor) SELECT nominations.nominee from nominations"
```


```python
conn.execute(insert_into_movies)
conn.execute(insert_into_actors)
```




    <sqlite3.Cursor at 0x1cefc3693b0>




```python
print(conn.execute("SELECT * FROM actors LIMIT 5").fetchall())
print(conn.execute("SELECT * FROM movies LIMIT 5").fetchall())
```

    [(1, 'Javier Bardem'), (2, 'Jeff Bridges'), (3, 'Jesse Eisenberg'), (4, 'Colin Firth'), (5, 'James Franco')]
    [(1, 'Biutiful'), (2, 'True Grit'), (3, 'The Social Network'), (4, "The King's Speech"), (5, '127 Hours')]
    


```python
q14 = "SELECT movie, nominee FROM nominations"
list_actors_movies = conn.execute(q14).fetchall()
list_actors_movies[0:5]
```




    [('Biutiful', 'Javier Bardem'),
     ('True Grit', 'Jeff Bridges'),
     ('The Social Network', 'Jesse Eisenberg'),
     ("The King's Speech", 'Colin Firth'),
     ('127 Hours', 'James Franco')]




```python
insert_query = '''
    INSERT INTO movies_actors(movie_id, actor_id)
    VALUES ((SELECT id FROM movies WHERE movie == ?), (SELECT id from actors where actor == ?))
'''
conn.executemany(insert_query, list_actors_movies)

print(conn.execute("SELECT * FROM movies_actors LIMIT 5").fetchall())
```

    [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5)]
    

---

#### Learning Summary


Python/SQL concepts explored: python+sqlite3, pandas, multiple tables, foreign keys, subqueries, populating new tables

Python functions and methods used: .cursor(), .connect(), .execute(), .fetchall(), .executemany() 

SQL statements used: PRAGMA, LIMIT, FROM, SELECT, INNER JOIN, DROP, ALTER, VALUES


The files used for this project can be found in my [GitHub repository](https://github.com/sengkchu/Dataquest-Guided-Projects-Solutions/tree/master/Guided%20Project_%20Creating%20relations%20in%20SQLite).
