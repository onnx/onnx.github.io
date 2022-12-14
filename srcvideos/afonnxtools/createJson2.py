#!/usr/bin/env python
# coding: utf-8

import json
import pandas as pd
import pathlib
import datetime
import os
import re


# In[2]:


def create_json_files(outputpath, df):

    # Index(['Title', 'Speaker', 'Description', 'Video_Url', 'related_urls', 'Date',
    # 'language', 'folder', 'youtubeid'],

    datetimestr = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outputdir = pathlib.Path("/home/andi/Andreas/onnxvideosrc/output/" + datetimestr)

    outputdir = outputpath 
    pathlib.Path.mkdir(outputdir, mode=0o777, parents=False, exist_ok=True)

    print(datetimestr)
    for index, row in df.iterrows():
        print(row["Title"], row["Speaker"])

        with open(templatefile, "r") as f:
            data = json.load(f)

        if row["Tags"] == "NaN":
            data.update({"description": ""})
        else:
            tags = row["Tags"].split(", ")
            # print(data['speakers']: )
            data.update({"Tags": tags})

        newspeaker = row["Speaker"].replace("\xa0", " ").split(", ")
        # print(data['speakers']: )
        data.update({"speakers": newspeaker})

        data.update({"title": row["Title"]})
        if row["Description"] == "NaN":
            data.update({"description": "NaN"})
        else:
            data.update({"description": row["Description"]})
        data.update({"duration": ""})
        data.update({"related_urls": ""})
        data.update({"copyright_text": "Needs to be clarifed"})
        data.update({"recorded": row["recorded"]})

        newthumburl = "https://i.ytimg.com/vi/" + row["youtubeid"] + "/hqdefault.jpg"
        # print(newthumburl)
        data.update({"thumbnail_url": newthumburl})

        data.get("videos")[0].update({"url": row["Video_Url"]})

        # print('videourl: ' + row['videos'])

        titlestr = row["Title"]
        # print(titlestr)

        new_titlestr = "".join(char for char in titlestr if char.isalnum())
        # print('newtitle: ' + str(new_titlestr))

        # ffprobe -i sample_5.mp4 -v quiet -show_entries format=duration -hide_banner -of default=noprint_wrappers=1:nokey=1

        pathlib.Path(outputdir.__str__() + os.sep + row["folder"] + os.sep + 'videos').mkdir(
            mode=0o777, parents=True, exist_ok=True
        )

        print(data)
        print(
            outputdir.__str__()
            + os.sep
            + row["folder"]
            + os.sep
            + "videos"
            + os.sep
            + new_titlestr
            + ".json"
        )

        with open(
            outputdir.__str__()
            + os.sep
            + row["folder"]
            + os.sep
            + "videos"
            + os.sep
            + new_titlestr
            + ".json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(data, f, ensure_ascii=False, indent=4)


# In[3]:


def extractyoutube(text):
    print(text)
    if "youtu.be" in text:
        m = re.search("(?<=youtu.be/).*", text)
        print(text)
        print(m)
        result = m.group(0)
        print(result)
    else:  # 'youtube.com' in text:
        print(text)
        # m = re.search('\?v=(.+?)', text)
        m = re.search("(?<=\?v=).*", text)

        print(text)
        print(m)
        result = m.group(0)

    print(result)
    return result



# This is available natively in pandas 0.25. So long as you have odfpy installed (conda install odfpy OR pip install odfpy) you can do


df = pd.read_excel("/home/andi/Andreas/onnxvideosrc/communityall2.ods", engine="odf")




df.columns


df["recorded"] = df["recorded"].astype("string")


df["Description"] = df["Description"].fillna("")


df.head(10)



df["youtubeid"] = df.Video_Url.apply(extractyoutube)



templatefile = r"/home/andi/Andreas/onnxvideosrc/template.json"
outputpath = "as"
create_json_files(outputpath, df)
