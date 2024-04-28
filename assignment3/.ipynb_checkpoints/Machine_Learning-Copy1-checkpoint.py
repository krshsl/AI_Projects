{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "7892ca62-4f40-4b6f-9e99-4aef31fc4061",
   "metadata": {
    "id": "7892ca62-4f40-4b6f-9e99-4aef31fc4061",
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import math\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import accuracy_score\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.tree import DecisionTreeRegressor\n",
    "from sklearn.metrics import mean_squared_error\n",
    "from sklearn.model_selection import cross_val_score\n",
    "from sklearn.metrics import accuracy_score"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "aa2e2386-8319-4d9e-b286-8465b73c6d10",
   "metadata": {},
   "source": [
    "Functions"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "76935c00-e549-45f7-b94a-8f5e3b2919f2",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def parse_tuple(cell):\n",
    "    parts = cell.strip('()').split(',')\n",
    "    return tuple(int(part.strip()) for part in parts)\n",
    "\n",
    "def clean_Bot_Move(row, df):\n",
    "    crew_cell = parse_tuple(row['Crew_Cell'])\n",
    "    if crew_cell[0] == 5 and crew_cell[1] == 5:\n",
    "        return None\n",
    "    else:\n",
    "        next_row_index = row.name + 1  # Get the index of the next row\n",
    "        if next_row_index < len(df):\n",
    "            return df.at[next_row_index, 'Bot_Cell']  # Return Bot_Cell value from the next row\n",
    "            \n",
    "def parse_coordinates(coord_str):\n",
    "    if coord_str:\n",
    "        x, y = map(int, coord_str.strip(\"()\").split(\",\"))\n",
    "        return x, y\n",
    "    else:\n",
    "        return None\n",
    "\n",
    "def calculate_direction(row):\n",
    "    bot_cell = parse_coordinates(row['Bot_Cell'])\n",
    "    bot_move = parse_coordinates(row['Bot_Move'])\n",
    "\n",
    "    if bot_cell and bot_move:\n",
    "        delta_x = bot_move[0] - bot_cell[0]\n",
    "        delta_y = bot_move[1] - bot_cell[1]\n",
    "\n",
    "        if delta_x == 0 and delta_y == 0:\n",
    "            return \"No movement\"\n",
    "        elif delta_x == 0:\n",
    "            return \"North\" if delta_y > 0 else \"South\"\n",
    "        elif delta_y == 0:\n",
    "            return \"East\" if delta_x > 0 else \"West\"\n",
    "        elif delta_x > 0:\n",
    "            return \"Northeast\" if delta_y > 0 else \"Southeast\"\n",
    "        else:\n",
    "            return \"Northwest\" if delta_y > 0 else \"Southwest\"\n",
    "    else:\n",
    "        return \"Invalid coordinates\"\n",
    "\n",
    "def map_coordinates_to_integer(row,celltype):\n",
    "    cell = parse_coordinates(row[celltype])\n",
    "    cols = 11\n",
    "    return cell[0] * cols + cell[1] + 1\n",
    "\n",
    "def parse_wall_coordinates(cell):\n",
    "    # Remove leading and trailing brackets and split by comma\n",
    "    cells = cell.strip('[]').split(',')\n",
    "    coordinates = []\n",
    "    for cell in cells:\n",
    "        # Extract coordinates from each cell, remove parentheses, and split by comma\n",
    "        parts = cell.strip('()').split(',')\n",
    "        # Convert parts to integers and create tuple\n",
    "        coordinate = tuple(int(part.strip()) for part in parts if part.strip().isdigit())\n",
    "        coordinates.append(coordinate)\n",
    "    return coordinates\n",
    "\n",
    "def encode_closed_cells(row):\n",
    "    # Convert string representation of Closed_cells to list of cell tuples\n",
    "    cells = parse_wall_coordinates(row['Closed_Cells'])\n",
    "    # Assign a unique identifier to each unique cell\n",
    "    unique_cells = set(cells)\n",
    "    cell_mapping = {cell: i for i, cell in enumerate(unique_cells)}\n",
    "    # Concatenate the identifiers of the cells to form a single encoded value\n",
    "    encoded_value = ''.join(str(cell_mapping[cell]) for cell in cells)\n",
    "    return encoded_value\n",
    "\n",
    "def lengthSquare(X, Y):\n",
    "    xDiff = X[0] - Y[0]\n",
    "    yDiff = X[1] - Y[1]\n",
    "    return xDiff * xDiff + yDiff * yDiff\n",
    "\n",
    "def getAngle(a, b):\n",
    "    c = (5, 5)\n",
    "    a2 = lengthSquare(a, c)\n",
    "    b2 = lengthSquare(b, c)\n",
    "    c2 = lengthSquare(a, b)\n",
    "    return math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2)))#(math.acos((a2 + b2 - c2) / (2 * math.sqrt(a2) * math.sqrt(b2))) * 180 / math.pi);\n",
    "\n",
    "def parse_angles(row):\n",
    "    crew_cell = parse_tuple(row['Crew_Cell'])\n",
    "    bot_cell = parse_tuple(row['Bot_Cell'])\n",
    "    return getAngle(bot_cell, crew_cell)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "31c03c41-4776-41d9-a642-5a08d4a3ccf2",
   "metadata": {
    "id": "31c03c41-4776-41d9-a642-5a08d4a3ccf2"
   },
   "source": [
    "READ AND PREPROCESS DATA"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "b9b80195-bf7e-42b6-bfcc-7ee4e2689279",
   "metadata": {
    "id": "b9b80195-bf7e-42b6-bfcc-7ee4e2689279",
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def process_data():\n",
    "    df =pd.read_csv(\"walloutput.csv\")\n",
    "    # Extract x, y, p, and q values from the existing columns\n",
    "    df['bot_x'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))\n",
    "    df['bot_y'] = df['Bot_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))\n",
    "    df['crew_x'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[0].strip('()')))\n",
    "    df['crew_y'] = df['Crew_Cell'].apply(lambda x: int(x.split(',')[1].strip('()')))\n",
    "\n",
    "# Calculate the distance\n",
    "    df['Distance_from_bot_to_crew'] = abs(df['bot_x'] - df['crew_x']) + abs(df['bot_y'] - df['crew_y'])\n",
    "    df['Distance_from_bot_to_teleport'] = abs(df['bot_x'] - 5) + abs(df['bot_y'] - 5)\n",
    "    df['Distance_from_crew_to_teleport'] = abs(5 - df['crew_x']) + abs(5 - df['crew_y'])\n",
    "\n",
    "#Drop the intermediate columns x, y, p, and q if needed\n",
    "    df.drop(['crew_x', 'crew_y', 'bot_x', 'bot_y'], axis=1, inplace=True)\n",
    "    df['Bot_Move'] = df['Bot_Cell'].shift(-1)\n",
    "    df['Bot_Move'] = df.apply(lambda row: clean_Bot_Move(row, df), axis=1)\n",
    "    df =df.dropna()\n",
    "    df['Direction_of_Bot'] = df.apply(lambda row: calculate_direction(row), axis=1)\n",
    "\n",
    "    df[\"Bot_Cell_Encoded\"] = df.apply(lambda row: map_coordinates_to_integer(row,\"Bot_Cell\"), axis=1)\n",
    "    df[\"Crew_Cell_Encoded\"] = df.apply(lambda row: map_coordinates_to_integer(row,\"Crew_Cell\"), axis=1)\n",
    "    df[\"Bot_Move_Encoded\"] = df.apply(lambda row: map_coordinates_to_integer(row,\"Bot_Move\"), axis=1)\n",
    "    df['Wall_Encoded_value'] = df.apply(encode_closed_cells, axis=1)\n",
    "\n",
    "    label_encoder = LabelEncoder()\n",
    "    label_encoded_df = df.copy()\n",
    "    if label_encoded_df[\"Direction_of_Bot\"].dtype == 'object':\n",
    "        label_encoded_df[\"Direction_of_Bot\"] = label_encoder.fit_transform(label_encoded_df[\"Direction_of_Bot\"])\n",
    "    \n",
    "    label_encoded_df = label_encoded_df.drop('Bot_Cell',axis =1)\n",
    "    label_encoded_df = label_encoded_df.drop('Crew_Cell',axis =1)\n",
    "    label_encoded_df = label_encoded_df.drop('Bot_Move',axis =1)\n",
    "    label_encoded_df = label_encoded_df.drop('Closed_Cells',axis =1)\n",
    "    label_encoded_df = label_encoded_df.drop('Walls',axis =1)\n",
    "    label_encoded_df = label_encoded_df.drop(\"Distance_from_bot_to_teleport\",axis=1)\n",
    "    label_encoded_df = label_encoded_df.drop(\"Distance_from_crew_to_teleport\",axis=1)\n",
    "    label_encoded_df = label_encoded_df.drop(\"Direction_of_Bot\",axis=1)\n",
    "\n",
    "    return label_encoded_df"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8fd883b8-8cda-4d59-9585-8adb1db66b42",
   "metadata": {
    "id": "8fd883b8-8cda-4d59-9585-8adb1db66b42"
   },
   "source": [
    "TRAIN TEST SPLIT"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "646ffb8f-93fc-455c-ab69-2282424043e1",
   "metadata": {
    "id": "646ffb8f-93fc-455c-ab69-2282424043e1",
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def train_data(data):\n",
    "    final_data = data.copy()\n",
    "    final_data = final_data.dropna()\n",
    "    X = final_data.drop('Bot_Move_Encoded', axis=1)\n",
    "    y = final_data['Bot_Move_Encoded']\n",
    "    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "    return X_train, X_test, y_train, y_test"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c94c1725-4f74-4038-846e-00f0e35d605b",
   "metadata": {
    "id": "c94c1725-4f74-4038-846e-00f0e35d605b"
   },
   "source": [
    "MODELS"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d1deaca1-c58b-4875-bce6-60e0768456af",
   "metadata": {
    "id": "d1deaca1-c58b-4875-bce6-60e0768456af"
   },
   "source": [
    "LINEAR REGRESSION"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "8afd5242-c635-402b-8f9d-4dfd34fb90bf",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "8afd5242-c635-402b-8f9d-4dfd34fb90bf",
    "outputId": "2adc92d6-f80c-4928-f6d7-9e732688cfab",
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def Linear_Regression_Model(X_train, X_test, y_train, y_test):\n",
    "    model = LinearRegression()\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    return y_test, y_pred, X_train,model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f91a563-7778-4b7c-9eb7-22f12b9cebe1",
   "metadata": {
    "id": "7f91a563-7778-4b7c-9eb7-22f12b9cebe1"
   },
   "source": [
    "DECISION TREE REGRESSOR"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "2b2d2a53-f525-425b-9e65-2706b9b60b6b",
   "metadata": {
    "colab": {
     "base_uri": "https://localhost:8080/"
    },
    "id": "2b2d2a53-f525-425b-9e65-2706b9b60b6b",
    "outputId": "c2d23abf-f18c-429a-fba7-29ae0237932b",
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def Decision_Tree_Regressor(X_train, X_test, y_train, y_test):\n",
    "    model = DecisionTreeRegressor()\n",
    "    model.fit(X_train, y_train)\n",
    "    y_pred = model.predict(X_test)\n",
    "    return y_test, y_pred, X_train,model"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b732175a-68a7-44ae-a285-7987236b3d13",
   "metadata": {},
   "source": [
    "Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "623fc7c0-a2cf-4745-aacd-d0fa667ff6fc",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def reg_metrics(y_test, y_pred, X_train):\n",
    "    from sklearn.metrics import mean_squared_error, r2_score \n",
    "\n",
    "    rmse = np.sqrt(mean_squared_error(y_test,y_pred))\n",
    "    r2 = r2_score(y_test,y_pred)\n",
    "    n = y_pred.shape[0]\n",
    "    k = X_train.shape[1]\n",
    "    adj_r_sq = 1 - (1 - r2)*(n-1)/(n-1-k)\n",
    "    print(\"rmse:\", rmse)\n",
    "    print(\"r2:\", r2)\n",
    "    print(\"adj_r_sq:\", adj_r_sq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "70875655-6640-407f-9b54-c55e26eaf867",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def create_model(n):\n",
    "    data  = process_data()\n",
    "    X_train, X_test, y_train, y_test = train_data(data)\n",
    "    if n == 1:\n",
    "        y_test, y_pred, X_train,model = Decision_Tree_Regressor(X_train, X_test, y_train, y_test)\n",
    "    else:\n",
    "        y_test, y_pred, X_train,model = Linear_Regression_Model(X_train, X_test, y_train, y_test)\n",
    "    print(reg_metrics(y_test, y_pred, X_train))\n",
    "    return model\n",
    "        \n",
    "        \n",
    "        "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "eeff31da-c925-469a-ae45-d871009c08c2",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "def predict_model(model,input):\n",
    "    Xnew = input\n",
    "    ynew = model.predict(Xnew)\n",
    "    return int(ynew)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "e54ca051-3b42-454f-b786-a6703728fe03",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "rmse: 4.590778803416515\n",
      "r2: 0.9795825925538456\n",
      "adj_r_sq: 0.9794704089964491\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "model1 = create_model(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "b00a591e-a839-471c-8304-23b296de9a9c",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "rmse: 5.933529516643243\n",
      "r2: 0.9658921712075741\n",
      "adj_r_sq: 0.9657047655548685\n",
      "None\n"
     ]
    }
   ],
   "source": [
    "model2 = create_model(2)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "79d2711d-1a48-48b3-8099-448e49dd8c38",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "Xnew = [[2,103,93,22101210]]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cbab736e-a257-4bc5-a2f4-f0438e9180b3",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "predicted_output = predict_model(model1,Xnew)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "75fe814e-8174-4d20-abfa-fec2b3276e50",
   "metadata": {
    "vscode": {
     "languageId": "python"
    }
   },
   "outputs": [],
   "source": [
    "print(predicted_output)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "2b206e99",
   "metadata": {},
   "source": [
    "SOLVE THE GRID"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "5b7460ca",
   "metadata": {
    "vscode": {
     "languageId": "r"
    }
   },
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'AI_proj3'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[39], line 1\u001b[0m\n\u001b[1;32m----> 1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mAI_proj3\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpy\u001b[39;00m\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'AI_proj3'"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "37f06491-f912-47db-ab47-1410766f1684",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "colab": {
   "provenance": []
  },
  "kernelspec": {
   "display_name": "R",
   "language": "R",
   "name": "ir"
  },
  "language_info": {
   "codemirror_mode": "r",
   "file_extension": ".r",
   "mimetype": "text/x-r-source",
   "name": "R",
   "pygments_lexer": "r",
   "version": "3.6.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
