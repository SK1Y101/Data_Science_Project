# Data_Science_Project
This is an implementation of a data science pipeline that predicts the outcome of a football match.


## High level relationships
- 14 leagues
 - data begining 1990
 - typically 380 games per season
  - 190 unique opponent combinations, as each team plays both an away and home against each other team
  - gives 20 teams per league (20 Choose 2 = 190)

formula for number of teams:

combinations of n, given a smaple size r = n! / (r! (n-r)!)

d = length of data

there are two teams playing, thus

d / 2 = n! / (2! (n-2)!)

Simplifies to
d = n (n-1)

## Trends
