#-------Set Parameters--------#
ticker = "BTC-USD"
unit_hours = 6
days_begin = 10
days_end = 10

end_date = datetime.utcnow() - timedelta(days_begin)
start_date = end_date - timedelta(days=days_end)
print(end_date, start_date)

#------Get Data-------#
train_data = yf.download(
    tickers=ticker,
    start=start_date.strftime("%Y-%m-%d"),
    end=end_date.strftime("%Y-%m-%d"),
    interval="15m",
    progress=False,
    group_by='ticker',
    auto_adjust=False
)
#-------Drop The NAme-------#
train_data.columns = train_data.columns.droplevel(0)
train_data.head()
