const AppError = require("../../utils/AppError");
const catchAsync = require("../../utils/catchAsync");
const JwtSession = require("../../models/mongodb/JwtSession");


exports.logout = catchAsync(async (req, res, next) => {
  const token = req.headers.authorization?.split(" ")[1];

  if (!token) {
    return next(new AppError("Unauthorized", 401));
  }

  await JwtSession.updateOne(
    { accessToken: token },
    { isValid: false }
  );

  res.status(200).json({
    success: true,
    message: "Logged out successfully"
  });
});