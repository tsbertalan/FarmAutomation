--
-- TelemRecSpec
--

TelemRecSpec = {
	MOD_NAME = g_currentModName
}

function TelemRecSpec.prerequisitesPresent(specializations)
	return true
end


-- These are functions which are called by the game and is linked to the new way
--  of how the vehicles handle specializations as to how to use them is anyones guess at the moment.
-- registerEvents
-- registerEventListeners
-- registerFunctions
-- registerOverwrittenFunctions

function TelemRecSpec.registerEventListeners(vehicleType)
	print("TelemRecSpec:registerEventListeners for vehicleType " .. tostring(vehicleType) .. ":")
	-- vehicleType is a table. Let's print its contents:
	for k,v in pairs(vehicleType) do
		print("  " .. tostring(k) .. " = " .. tostring(v))
	end
	
	SpecializationUtil.registerEventListener(vehicleType, "onPreLoad", TelemRecSpec)
	SpecializationUtil.registerEventListener(vehicleType, "onLoad", TelemRecSpec)
	SpecializationUtil.registerEventListener(vehicleType, "onEnterVehicle", TelemRecSpec)
	SpecializationUtil.registerEventListener(vehicleType, "onUpdate", TelemRecSpec)
end

function TelemRecSpec:onPreLoad(savegame)
	-- print("TelemRecSpec:onPreLoad function in vehicle")
end

function TelemRecSpec:onLoad(savegame)
	self.spec_TelemRecSpec = self["spec_" .. TelemRecSpec.MOD_NAME .. ".TelemRecSpec"]

	-- print("self.spec_TelemRecSpec " .. tostring(self.spec_TelemRecSpec))
end

function TelemRecSpec:onEnterVehicle(isControlling, playerStyle, farmId, playerIndex)
	print("TelemRecSpec:onEnterVehicle(isControlling=" .. tostring(isControlling) .. ", playerStyle=" .. tostring(playerStyle) .. ", farmId=" .. tostring(farmId) .. ", playerIndex=" .. tostring(playerIndex) .. ")")	
	print("Telemetry vehicle name is  ".. tostring(self:getName()))
end

function TelemRecSpec:onUpdate(dt)
	-- for i, component in pairs(self.components) do
	-- 	local vx, vy, vz = getLinearVelocity(component.node)
	-- 	local velx, vely, velz = getAngularVelocity(component.node)
	-- 	-- print("Telemetry vehicle velocity is  ".. tostring(vx) .. ", " .. tostring(vy) .. ", " .. tostring(vz))
	-- 	-- print("Telemetry vehicle angular velocity is  ".. tostring(velx) .. ", " .. tostring(vely) .. ", " .. tostring(velz))
	-- 	local PosX, PosY, PosZ = getWorldTranslation(component.node)
	-- 	local quatX, quatY, quatZ, quatW = getWorldQuaternion(component.node)
	-- 	-- print("Telemetry vehicle rotation is  ".. tostring(quatX) .. ", " .. tostring(quatY) .. ", " .. tostring(quatZ) .. ", " .. tostring(quatW))
	-- 	-- print("Telemetry vehicle position is  ".. tostring(PosX) .. ", " .. tostring(PosY) .. ", " .. tostring(PosZ))
	-- 	local torque = getMotorTorque(component.node)
	-- 	-- print("Telemetry vehicle torque is  ".. tostring(torque))
	-- end
end