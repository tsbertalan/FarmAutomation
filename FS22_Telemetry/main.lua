--
-- Mod: FS22_Telemetry
--
-- Author: Tom Bertalan
-- email: fs@tombertalan.com
-- Date: 2023-05-06

-- ####################################################################################################

local telem_main

local function isActive()
    print("FS22_Telemetry/main.lua: isActive() called.")
    return telem_main ~= nil
end

local function registerActionEvents(mission)
    print("FS22_Telemetry/main.lua: registerActionEvents() called.")
    if isActive() then
        telem_main:onRegisterActionEvents(mission, mission.inputManager)
    end
end

local function unregisterActionEvents(mission)
    if isActive() then
        telem_main:onUnregisterActionEvents(mission, mission.inputManager)
    end
end

local function init()

    print("Loading FS22_Telemetry/main.lua...")
    print("version bump 2")
    source(Utils.getFilename("scripts/TelemMainObj.lua", g_currentModDirectory))
    TelemRegisterActionEvents()
    
    telem_main = TelemetryMain.new(g_server ~= nil, g_client ~= nil)
    
    FSBaseMission.registerActionEvents = Utils.appendedFunction(FSBaseMission.registerActionEvents, registerActionEvents)
    BaseMission.unregisterActionEvents = Utils.appendedFunction(BaseMission.unregisterActionEvents, unregisterActionEvents)

    print("FS22_Telemetry/register.lua loaded.")

end


init()