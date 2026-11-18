"""OpenCV main menu, rig settings and recording selection."""
import json

import cv2
import numpy as np

from .display import Window, label
from .experiments import report
from .settings import RECORDINGS, load, save, validate


def edit_settings(config, path):
    window = Window("Rig settings")
    keys = list(config)
    fields = [json.dumps(config[k]) for k in keys]
    selected, replace = 0, True
    message = "Click a field to replace its value. Tab moves down. Enter saves."
    try:
        while True:
            image = np.full((34*len(keys), 660, 3), 34, np.uint8)
            for i,(key,value) in enumerate(zip(keys,fields)):
                if i == selected:
                    image[i*34:(i+1)*34] = 60
                label(image,key.replace("_"," "),(10,i*34+22))
                label(image,value,(240,i*34+22))
            button, click, key = window.poll(image, ["Save", "Cancel"], message)
            if button == "Cancel" or key == 27:
                return
            if button == "Save" or key in (10,13):
                try:
                    updated = {name:json.loads(value) for name,value in zip(keys,fields)}
                    validate(updated)
                    save(path,updated)
                    config.update(updated)
                    return
                except (ValueError,TypeError,OSError) as exc:
                    message = str(exc)
            elif click and click[1]//34 < len(keys):
                selected, replace = click[1]//34, True
            elif key == 9:
                selected, replace = (selected+1)%len(keys), True
            elif key in (8,127):
                fields[selected] = "" if replace else fields[selected][:-1]
                replace = False
            elif 32 <= key < 127:
                fields[selected] = ("" if replace else fields[selected])+chr(key)
                replace = False
    finally:
        window.close()


def select_recordings():
    window = Window("Select recordings")
    folder = RECORDINGS if RECORDINGS.is_dir() else RECORDINGS.parent
    selected = set()
    page = 0
    try:
        while True:
            entries = sorted(folder.iterdir(), key=lambda p:(not p.is_dir(),p.name.lower()))
            entries = [p for p in entries if p.is_dir() or p.suffix.lower()==".csv"]
            image = np.full((460,650,3), 34, np.uint8)
            label(image,str(folder),(8,23))
            for i,p in enumerate(entries[page:page+14]):
                prefix = "[dir] " if p.is_dir() else ("[x] " if p in selected else "[ ] ")
                label(image,prefix+p.name,(12,55+i*28))
            button,click,key = window.poll(image, ["Analyze selected","Previous page","Next page","Parent folder","Cancel"],
                                           f"{len(selected)} selected. Click CSV files to include them.")
            if key == 27 or button == "Cancel":
                return []
            if button == "Analyze selected" and selected:
                return sorted(selected)
            if button == "Previous page":
                page = max(0,page-14)
            elif button == "Next page" and page+14 < len(entries):
                page += 14
            elif button == "Parent folder":
                folder,page = folder.parent,0
            elif click and 0 <= (row := (click[1]-34)//28) < min(14,len(entries)-page):
                entry = entries[page+row]
                if entry.is_dir():
                    folder,page = entry,0
                else:
                    selected.symmetric_difference_update([entry])
    finally:
        window.close()


def launch(path, initial_mode=None):
    config = load(path)
    results = []
    mode = initial_mode
    window = Window("Stewart platform")
    try:
        while True:
            image = np.full((240,800,3), 34, np.uint8)
            label(image,"Stewart platform ball balancer",(12,32))
            for i,line in enumerate(results):
                label(image,line,(12,70+24*i))
            button,_,key = window.poll(image, ["Rig settings","Live balancer","Simulate","Analyze recordings","Exit"])
            if key == 27 or button == "Exit":
                return
            try:
                if button == "Rig settings":
                    edit_settings(config,path)
                elif button == "Live balancer" or mode:
                    from .live import LiveBalancer
                    initial, mode = mode, None
                    LiveBalancer(config,path).run(initial)
                elif button == "Simulate":
                    results = report(show=True)
                elif button == "Analyze recordings":
                    paths = select_recordings()
                    if paths:
                        results = report(paths,show=True)
            except (OSError,ValueError,RuntimeError,ImportError,KeyError,cv2.error) as exc:
                results = [str(exc)]
    finally:
        window.close()
