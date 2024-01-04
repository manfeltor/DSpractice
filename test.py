import quandl

df = quandl.get('WIKI/GOOGL')

print(df.iloc[-1].name)