import pandas as pd
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import gspread_dataframe as gd

dt = {
                "overall_rating" : [2] ,
                "recommendation" : [4],
              }
df = pd.DataFrame(data=dt)

scope = ["https://spreadsheets.google.com/feeds",'https://www.googleapis.com/auth/spreadsheets',"https://www.googleapis.com/auth/drive.file","https://www.googleapis.com/auth/drive"]
credentials = ServiceAccountCredentials.from_json_keyfile_name(
    './creds.json', scope)
gc = gspread.authorize(credentials)
ws = gc.open("SIGABRT_FEEDBACK").worksheet("Master")
existing = pd.DataFrame(ws.get_all_records())
updated = existing.append(df)
gd.set_with_dataframe(ws, updated)

