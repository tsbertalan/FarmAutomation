TelemetrySaver = {}

function TelemetrySaver:new(vehicle)
    print("TelemetrySaver:new called with vehicle=" .. vehicle)
    local o = {}
    self.__index = self
    o.vehicle = vehicle
    return o
end


function TelemetrySaver.prerequisitesPresent(specializations)
    print("TelemetrySaver.prerequisitesPresent called with specializations=" .. specializations)
    return true
end

function TelemetrySaver.registerEventListeners(vehicleType)
    print("TelemetrySaver.registerEventListeners")
    for _, n in pairs(
        {
            "onEnterVehicle"
        }
    ) do
        SpecializationUtil.registerEventListener(vehicleType, n, TelemetrySaver)
    end
end

function TelemetrySaver:onEnterVehicle(isControlling)
    print("TelemetrySaver:onEnterVehicle called with isControlling=" .. isControlling)
end
