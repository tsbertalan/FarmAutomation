--
-- Mod: FS22_Telemetry
--
-- Author: Tom Bertalan
-- email: fs@tombertalan.com
-- Date: 2023-05-06

-- ####################################################################################################

local telem_main

local function isActive()
    return telem_main ~= nil
end

local function registerActionEvents(mission)
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

    source(Utils.getFilename("scripts/TelemMainObj.lua", g_currentModDirectory))
    source(Utils.getFilename("scripts/TelemetrySaverSpecialization.lua", g_currentModDirectory))
    
    telem_main = TelemetryMain.new(g_server ~= nil, g_client ~= nil)
    
    FSBaseMission.registerActionEvents = Utils.appendedFunction(FSBaseMission.registerActionEvents, registerActionEvents)
    BaseMission.unregisterActionEvents = Utils.appendedFunction(BaseMission.unregisterActionEvents, unregisterActionEvents)


    if g_specializationManager:getSpecializationByName("TelemetrySaver") == nil then
        -- name, className, filename, customEnvironment
        g_specializationManager:addSpecialization("TelemetrySaver", "TelemetrySaver", Utils.getFilename("scripts/TelemetrySaverSpecialization.lua", g_currentModDirectory), nil)
    end

end


init()