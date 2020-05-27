df.drop(df.loc[df['line_race']==0].index, inplace=True) 
df = df.drop(df[df['line_race']==0].index)


You can use groupby on index after sorting the df by valu.

df.sort_values(by='valu', ascending=False).groupby(level=0).first()
Out[1277]: 
           is_avail   valu data_source
2015-08-07    False  0.582    source_b
2015-08-23    False  0.296    source_a
2015-09-08    False  0.433    source_a
2015-10-01     True  0.169    source_b

Using drop_duplicates with keep='last'

df.rename_axis('date').reset_index() \
    .sort_values(['date', 'valu']) \
    .drop_duplicates('date', keep='last') \
    .set_index('date').rename_axis(df.index.name)

           is_avail   valu data_source
2015-08-07    False  0.582    source_b
2015-08-23    False  0.296    source_a
2015-09-08    False  0.433    source_a
2015-10-01     True  0.169    source_b


  
  