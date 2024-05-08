import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

def parse_tuple(cell):
    parts = cell.strip('()').split(',')
    return tuple(int(part.strip()) for part in parts)

def clean_Bot_Move(row, df):
    crew_cell = parse_tuple(row['Crew_Cell'])
    if crew_cell[0] == 5 and crew_cell[1] == 5:
        return None
    else:
        next_row_index = row.name + 1  # Get the index of the next row
        if next_row_index < len(df):
            return df.at[next_row_index, 'Bot_Cell']  # Return Bot_Cell value from the next row
            
def parse_coordinates(coord_str):
    if coord_str:
        x, y = map(int, coord_str.strip("()").split(","))
        return x, y
    else:
        return None

def calculate_direction(row):
    bot_cell = parse_coordinates(row['Bot_Cell'])
    bot_move = parse_coordinates(row['Bot_Move'])

    if bot_cell and bot_move:
        delta_x = bot_move[0] - bot_cell[0]
        delta_y = bot_move[1] - bot_cell[1]

        if delta_x == 0 and delta_y == 0:
            return "No movement"
        elif delta_x == 0:
            return "North" if delta_y > 0 else "South"
        elif delta_y == 0:
            return "East" if delta_x > 0 else "West"
        elif delta_x > 0:
            return "Northeast" if delta_y > 0 else "Southeast"
        else:
            return "Northwest" if delta_y > 0 else "Southwest"
    else:
        return "Invalid coordinates"

def map_coordinates_to_integer(row,celltype):
    cell = parse_coordinates(row[celltype])
    cols = 11
    return cell[0] * cols + cell[1] + 1

def parse_wall_coordinates(cell):
    # Remove leading and trailing brackets and split by comma
    cells = cell.strip('[]').split(',')
    coordinates = []
    for cell in cells:
        # Extract coordinates from each cell, remove parentheses, and split by comma
        parts = cell.strip('()').split(',')
        # Convert parts to integers and create tuple
        coordinate = tuple(int(part.strip()) for part in parts if part.strip().isdigit())
        coordinates.append(coordinate)
    return coordinates

def encode_closed_cells(row):
    # Convert string representation of Closed_cells to list of cell tuples
    cells = parse_wall_coordinates(row['Closed_Cells'])
    # Assign a unique identifier to each unique cell
    unique_cells = set(cells)
    cell_mapping = {cell: i for i, cell in enumerate(unique_cells)}
    # Concatenate the identifiers of the cells to form a single encoded value
    encoded_value = ''.join(str(cell_mapping[cell]) for cell in cells)
    return encoded_value

def lengthSquare(X, Y):
    xDiff = X[0] - Y[0]
    yDiff = X[1] - Y[1]
    return xDiff * xDiff + yDiff * yDiff

def getAngle(a, b):
    c = (5, 5)
    a2 = lengthSquare(a, c)
    b2 = lengthSquare(b, c)
    c2 = lengthSquare(a, b)
    return math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2)))#(math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2))) * 180 / math.pi);

def parse_angles(row):
    crew_cell = parse_tuple(row['Crew_Cell'])
    bot_cell = parse_tuple(row['Bot_Cell'])
    return getAngle(bot_cell, crew_cell)

def process_data():
    df =pd.read_csv("walloutput.csv")
    # Extract x, y, p, and q values from the existing columns
    df['bot_x'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['bot_y'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))
    df['crew_x'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['crew_y'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))

# Calculate the distance
    df['Distance_from_bot_to_crew'] = abs(df['bot_x'] - df['crew_x']) + abs(df['bot_y'] - df['crew_y'])
    df['Distance_from_bot_to_teleport'] = abs(df['bot_x'] - 5) + abs(df['bot_y'] - 5)
    df['Distance_from_crew_to_teleport'] = abs(5 - df['crew_x']) + abs(5 - df['crew_y'])

#Drop the intermediate columns x, y, p, and q if needed
    df.drop(['crew_x', 'crew_y', 'bot_x', 'bot_y'], axis=1, inplace=True)
    df['Bot_Move'] = df['Bot_Cell'].shift(-1)
    df['Bot_Move'] = df.apply(lambda row: clean_Bot_Move(row, df), axis=1)
    df =df.dropna()
    df['Direction_of_Bot'] = df.apply(lambda row: calculate_direction(row), axis=1)

    df["Bot_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Cell"), axis=1)
    df["Crew_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Crew_Cell"), axis=1)
    df["Bot_Move_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Move"), axis=1)
    df['Wall_Encoded_value'] = df.apply(encode_closed_cells, axis=1)


def parse_tuple(cell):
    parts = cell.strip('()').split(',')
    return tuple(int(part.strip()) for part in parts)

def clean_Bot_Move(row, df):
    crew_cell = parse_tuple(row['Crew_Cell'])
    if crew_cell[0] == 5 and crew_cell[1] == 5:
        return None
    else:
        next_row_index = row.name + 1  # Get the index of the next row
        if next_row_index < len(df):
            return df.at[next_row_index, 'Bot_Cell']  # Return Bot_Cell value from the next row

def parse_coordinates(coord_str):
    if coord_str:
        x, y = map(int, coord_str.strip("()").split(","))
        return x, y
    else:
        return None

def map_coordinates_to_integer(row,celltype):
    cell = parse_coordinates(row[celltype])
    cols = 11
    return cell[0] * cols + cell[1] + 1

def parse_wall_coordinates(cell):
    # Remove leading and trailing brackets and split by comma
    cells = cell.strip('[]').split(',')
    coordinates = []
    for cell in cells:
        # Extract coordinates from each cell, remove parentheses, and split by comma
        parts = cell.strip('()').split(',')
        # Convert parts to integers and create tuple
        coordinate = tuple(int(part.strip()) for part in parts if part.strip().isdigit())
        coordinates.append(coordinate)
    return coordinates

def encode_closed_cells(row):
    # Convert string representation of Closed_cells to list of cell tuples
    cells = parse_wall_coordinates(row['Closed_Cells'])
    # Assign a unique identifier to each unique cell
    unique_cells = set(cells)
    cell_mapping = {cell: i for i, cell in enumerate(unique_cells)}
    # Concatenate the identifiers of the cells to form a single encoded value
    encoded_value = ''.join(str(cell_mapping[cell]) for cell in cells)
    return encoded_value

def process_data():
    df =pd.read_csv("walloutput.csv")
    # Extract x, y, p, and q values from the existing columns
    df['bot_x'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['bot_y'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))
    df['crew_x'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))
    df['crew_y'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))

    df['Distance_from_bot_to_crew'] = abs(df['bot_x'] - df['crew_x']) + abs(df['bot_y'] - df['crew_y'])
    df["Bot_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Cell"), axis=1)
    df["Crew_Cell_Encoded"] = df.apply(lambda row: map_coordinates_to_integer(row,"Crew_Cell"), axis=1)
    df['Bot_Move_Encoded'] = df.apply(lambda row: map_coordinates_to_integer(row,"Bot_Move"), axis=1)
    df['Wall_Encoded_value'] = df.apply(encode_closed_cells, axis=1)

    df =  df.drop('Bot_Cell',axis = 1)
    df =  df.drop('Crew_Cell',axis = 1)
    df =  df.drop('Closed_Cells',axis = 1)
    df =  df.drop('Walls',axis =1)

    return df

def train_data(data):
    final_data = data.copy()
    final_data = final_data.dropna()
    X = final_data.drop('Bot_Move_Encoded', axis=1)
    y = final_data['Bot_Move_Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def Decision_Tree_Regressor(X_train, X_test, y_train, y_test):
    model = DecisionTreeRegressor()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_test, y_pred, X_train,model

def reg_metrics(y_test, y_pred, X_train):
    from sklearn.metrics import mean_squared_error, r2_score 

    rmse = np.sqrt(mean_squared_error(y_test,y_pred))
    r2 = r2_score(y_test,y_pred)
    n = y_pred.shape[0]
    k = X_train.shape[1]
    adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)
    print("rmse:", rmse)
    print("r2:", r2)
    print("adj_r_sq:", adj_r_sq)

def create_model(n):
    data  = process_data()
    X_train, X_test, y_train, y_test = train_data(data)
    print(X_train)
    y_test, y_pred, X_train,model = Decision_Tree_Regressor(X_train, X_test, y_train, y_test)
    return model

def predict_model(model,input):
    Xnew = input
    ynew = model.predict(Xnew)
    return int(ynew)

model = create_model(1)