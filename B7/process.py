import pandas as pd

# Đọc dữ liệu từ CSV
df = pd.read_csv('dataset.csv', parse_dates=['timestamp'])

# Đặt cột 'timestamp' làm chỉ mục
df.set_index('timestamp', inplace=True)

# Tổng hợp lượt mention của Apple theo ngày
daily_mentions = df['Apple'].resample('D').sum()

# Xuất kết quả ra console
print(daily_mentions)

# Lưu kết quả vào file CSV nếu cần
daily_mentions.to_csv('daily_mentions_apple.csv')
