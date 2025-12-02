-- local socket = require("socket")

-- function socketDemo()

--     -- Define the host and port to connect to
--     local host = "127.0.0.1"
--     local port = 12345

--     -- Create a client socket
--     local client = assert(socket.tcp())

--     -- Connect to the server
--     client:connect(host, port)

--     -- Send data to the server
--     client:send("Hello from Lua!")

--     -- Close the connection
--     client:close()

-- end

function TelemToggle()
    print("TelemToggle function called.")
    -- socketDemo()
end

TelemetryMain = {};
local TelemetryMain_mt = Class(TelemetryMain);

function TelemetryMain.new(isServer, isClient)
    -- Construct the global state manager object for this plugin.
    local self = setmetatable({}, TelemetryMain_mt)  -- god this language is trash
    g_inputBinding:registerActionEvent(InputAction.TelemToggle)
    print("======== TelemetryMain.new called. =========")
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
    print("TelemetryMain:onUnregisterActionEvents called.")
end

function TelemetryMain:update(dt)
    -- I guess this is nothing. Probably should be deleted.
    print("TelemetryMain:update called with dt=" .. dt)
end


