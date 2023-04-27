# Journal
## [[2023-04-19]]

### Hello World-ing
The FS19 version of Courseplay hasn't had a commit for since Christmas Eve 2021, so I bought FS22 (which was on a small "sale" today anyway).

I can just clone the [courseplay](https://github.com/Courseplay/Courseplay_FS22) [(Archived)](https://web.archive.org/web/20230419/https://github.com/Courseplay/Courseplay_FS22) and [autodrive](https://github.com/Stephan-S/FS22_AutoDrive) [(Archived)](https://web.archive.org/web/20230419/https://github.com/Stephan-S/FS22_AutoDrive) repos, modify the lua code, and have an "install" python script that copies the clones into `C:\Users\tsbertalan\Documents\My Games\FarmingSimulator2022\mods\`.

If I launch the game with the flags `-cheats -autoConnectDebugger -autoStartSavegameId 1`, and also do this in the `game.xml` a level up from `mods`:
```xml
<logging>
    <file enable="true" filename="log.txt"/>
    <console enable="true"/>
</logging>
<development>
    <controls>true</controls>
    <openDevConsole onWarnings="true" onErrors="true"/>
</development>
<startMode>1</startMode>
```
then the grave key opens a log (same output as `log.txt`, so whatever), but, more importantly, the game loads straight into save 1, and "quitting" the save just reloads it *with any lua script changes that might have happened on disk.* So I can run my `install.py` with Ctrl+Shift+B from vsc using this `tasks.json`:
```json
{
    // See https://go.microsoft.com/fwlink/?LinkId=733558
    // for the documentation about the tasks.json format
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Build",
            "type": "shell",
            "command": "C:/Users/tsbertalan/anaconda3/envs/pytorch3916/python.exe install.py",
            "group": {
                "kind": "build",
                "isDefault": true
            }
        }
    ]
}
```
then Esc, q, backspace to quit the save, then enter to accept the load, then my code is updated.

Unfortunately, the game developer does supply a [debugger/IDE](https://gdn.giants-software.com/debugger.php), but I can't get it to connect to the game process. Could be worth asking around.

### Data Exfiltration
Thanks to the Lua extension in VSC from user sumneko, I can F12 my way to the superclass constructor `AIDriveStrategyFieldWorkCourse.new`. Thanks to Copilot, I can speak Lua well enough to
```lua
function AIDriveStrategyFieldWorkCourse.new(customMt)
   -- We'll save telemetry data to a file as we drive:
    self.telemtry_file = io.open("telemetry.csv", "w")

    -- Write the column headers: dt, vX, vY, vZ, gx, gz, moveForwards, maxSpeed
    print('Opening telemetry file with columns: dt,vX,vY,vZ,gx,gz,moveForwards,maxSpeed\n')
    self.telemtry_file:write("dt,vX,vY,vZ,gx,gz,moveForwards,maxSpeed\n")
    
    -- ...
end

function AIDriveStrategyFieldWorkCourse:getDriveData(dt, vX, vY, vZ)
    -- ... generate gx, gz here ...
    -- Save the data to the telemtry file.
    self.telemtry_file:write(string.format("%f,%f,%f,%f,%f,%f,%s,%f\n", dt, vX, vY, vZ, gx, gz, tostring(moveForwards), self.maxSpeed))
    -- ...
end
```

And voila, I have a CSV file.

![telemetry_2023-04-19-16-39.csv](./Farm%20Automation.assets/telemetry_2023-04-19-16-39.csv.png)
That was all *so* much easier than in GTAV with C++ compilation cycles and those terrible (but brilliant) ScriptHookV memory hacks to get data out and commands in.

Next I'll need to
- [ ] find some additional useful data (such as pose, motor rpm, gear, torque, ...)
- [ ] Consider using sockets or some genuine pub-sub framework instead of a CSV dump.
