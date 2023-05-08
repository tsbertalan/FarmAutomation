function TelemToggle()
    print("TelemToggle function called.")
end

TelemetryMain = {};
local TelemetryMain_mt = Class(TelemetryMain);

function TelemetryMain.new(isServer, isClient)
    -- Construct the global state manager object for this plugin.
    local self = setmetatable({}, TelemetryMain_mt)  -- god this language is trash
    g_inputBinding:registerActionEvent(InputAction.TelemToggle)
    return self
end

function TelemetryMain:onRegisterActionEvents(mission, inputManager)
    -- Register actions that should happen in response to the configured input events (e.g. key presses).
    local firstThing, eventId = inputManager:registerActionEvent(
        InputAction.TelemToggle, self, TelemToggle, false, true, false, true
    )
end

function TelemetryMain:onUnregisterActionEvents(mission, inputManager)
    -- Unregister those same events.
    inputManager:removeActionEventsByTarget(self)
end

function TelemetryMain:update(dt)
    -- I guess this is nothing. Probably should be deleted.
    print("TelemetryMain:update called with dt=" .. dt)
end


