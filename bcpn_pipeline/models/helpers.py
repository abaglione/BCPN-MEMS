import csv
import json
import os
import aiofiles
from aiocsv import AsyncDictWriter

async def to_csv_async(df, fp, mode):
    mode = 'w' if os.path.exists(fp) else 'a' # Set up mode
    header = list(df.columns)
    df_json = df.to_json(orient="records")
    rows = json.loads(df_json)

    # dict writing, all quoted, "NULL" for missing fields
    async with aiofiles.open(fp, mode=mode, encoding="utf-8", newline='') as afp:
        writer = AsyncDictWriter(afp, header, restval="NULL", quoting=csv.QUOTE_ALL)
        if mode == 'w':
            await writer.writeheader()
        await writer.writerows(rows)