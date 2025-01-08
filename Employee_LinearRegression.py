import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error 


# Đọc dữ liệu từ file CSV
data = pd.read_csv('E:\code\HTKDTM\Data\Employee_data.csv')

# Tính tỷ lệ phần trăm cho mỗi mục
data['food_ratio'] = data['food_cost'] / data['total']
data['rent_ratio'] = data['rent'] / data['total']
data['other_cost_ratio'] = data['other_cost'] / data['total']
data['savings_ratio'] = data['savings'] / data['total']

# Chia dữ liệu thành các biến độc lập (X) và biến phụ thuộc (y)
X = data[['food_cost', 'rent', 'other_cost', 'savings']]
y = data[['food_ratio', 'rent_ratio', 'other_cost_ratio', 'savings_ratio']]

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tạo mô hình hồi quy tuyến tính
model = LinearRegression()

# Huấn luyện mô hình
model.fit(X_train, y_train)
# Nhập tổng tiền từ người dùng
total_amount = float(input("Nhập tổng số tiền: "))

# Dự đoán tỷ lệ phần trăm cho mỗi mục
predicted_ratios = model.predict([[total_amount, 0, 0, 0]])[0]

# Kiểm tra và điều chỉnh giá trị dự đoán
predicted_ratios = np.maximum(predicted_ratios, 0)  # Đảm bảo tất cả giá trị đều không âm

# Tính tỷ lệ phần trăm còn lại cho mục tiết kiệm
predicted_savings_ratio = 1 - np.sum(predicted_ratios[:3])

# Điều chỉnh lại tỷ lệ phần trăm nếu tổng lớn hơn 1
if np.sum(predicted_ratios) > 1:
    predicted_ratios /= np.sum(predicted_ratios)
    predicted_savings_ratio = 1 - np.sum(predicted_ratios[:3])

# Tính số tiền cho mỗi mục
predicted_food = round(predicted_ratios[0] * total_amount / 1000) * 1000
predicted_rent = round(predicted_ratios[1] * total_amount / 1000) * 1000
predicted_other = round(predicted_ratios[2] * total_amount / 1000) * 1000
predicted_savings = round(predicted_savings_ratio * total_amount / 1000) * 1000

# In kết quả với định dạng số có dấu cách sau mỗi 3 chữ số
print("Phân bổ chi tiêu dự kiến:")
print("Ăn uống:", "{:,}".format(predicted_food))
print("Nhà ở:", "{:,}".format(predicted_rent))
print("Chi phí khác:", "{:,}".format(predicted_other))
print("Tiết kiệm:", "{:,}".format(predicted_savings))

# Kiểm tra lại tổng
total_predicted = predicted_food + predicted_rent + predicted_other + predicted_savings
print("Tổng chi tiêu dự đoán:", "{:,}".format(total_predicted))

# Dự đoán trên tập kiểm tra
y_pred = model.predict(X_test)

# Đánh giá mô hình
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mô hình có khả năng dự đoán {r2*100:.2f}%")
print(f"Các chỉ số đánh giá:")
print("Mean Squared Error:", mse)
print("Root Mean Squared Error:", rmse)
print("Mean Absolute Error:", mae)