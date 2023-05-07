function TelemRegisterActionEvents()
    -- print the action key binding from InputAction.TelemToggle
    print("TelemToggle: " .. InputAction.TelemToggle)
end

function TelemToggle()
    -- print the action key binding from InputAction.TelemToggle
    print("TelemToggle function called.")
end

TelemetryMain = {};
local TelemetryMain_mt = Class(TelemetryMain);

function TelemetryMain.new(isServer, isClient)
    local self = setmetatable({}, TelemetryMain_mt)  -- god this language is trash
    g_inputBinding:registerActionEvent(InputAction.TelemToggle)
    return self
end

function TelemetryMain:onRegisterActionEvents(mission, inputManager)
    local firstThing, eventId = inputManager:registerActionEvent(
        InputAction.TelemToggle, self, TelemToggle, false, true, false, true
    )
end

function TelemetryMain:update(dt)
    print("TelemetryMain:update called with dt=" .. dt)
end


