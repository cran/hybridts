#' Hybrid ARIMA WARIMA Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param p An integer indicating the maximum order of AR process. Default is 5.
#' @param q An integer indicating the maximum order of MA process. Default is 5.
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast stats WaveletArima
#'
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE) for the forecast.
#'
#' @export
#'
#' @references \itemize{
#' \item Chakraborty, T., & Ghosh, I. (2020). Real-time forecasts and risk assessment of novel
#' coronavirus (COVID-19) cases: A data-driven analysis. Chaos, Solitons & Fractals, 135, 109850.
#'
#' \item Chakraborty, T., Ghosh, I., Mahajan, T., & Arora, T. (2022). Nowcasting of COVID-19 confirmed
#' cases: Foundations, trends, and challenges. Modeling, Control and Drug Development for COVID-19
#' Outbreak Prevention, 1023-1064.}
#'
#' @examples
#' arima_warima(y = datasets::lynx, n = 3)
#'
arima_warima = function(y, n, p = 5, q = 5, PI = FALSE, ret_fit = FALSE){
  fitARIMA = forecast::auto.arima(y)
  predARIMA = forecast::forecast(fitARIMA,h=n)
  fit_res_wbf = WaveletArima::WaveletFittingarma(fitARIMA$residuals, Waveletlevels = floor(log(length(y))),
                                                 boundary = 'periodic', FastFlag = TRUE, MaxARParam = p,
                                                 MaxMAParam = q, NForecast = n)
  fit_arima_wbf=predARIMA$fitted+fit_res_wbf$FinalPrediction
  pred_arima_wbf=predARIMA$mean+fit_res_wbf$Finalforecast
  if (isTRUE(PI)){
    upper = pred_arima_wbf + 1.5*stats::sd(pred_arima_wbf)
    lower = pred_arima_wbf - 1.5*stats::sd(pred_arima_wbf)
    forecast = list("Fitted" = fit_arima_wbf,
                    "Forecast" = pred_arima_wbf,
                    "Lower Interval" = lower,
                    "Upper Interval" = upper)

  }else{
    forecast = list("Fitted" = fit_arima_wbf, "Forecast" = pred_arima_wbf)
  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Hybrid ARIMA ARNN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE)
#' for the forecast.
#'
#' @export
#'
#' @references \itemize{
#' \item Chakraborty, T., Chattopadhyay, S., & Ghosh, I. (2019). Forecasting dengue epidemics using a
#' hybrid methodology. Physica A: Statistical Mechanics and its Applications, 527, 121266.
#'
#' \item Chakraborty, T., Ghosh, I., Mahajan, T., & Arora, T. (2022). Nowcasting of COVID-19 confirmed
#' cases: Foundations, trends, and challenges. Modeling, Control and Drug Development for COVID-19
#' Outbreak Prevention, 1023-1064.}
#'
#' @examples
#' arima_arnn(y = datasets::lynx, n = 3)
#'
#'
arima_arnn = function(y, n, PI = FALSE, ret_fit = FALSE){
  fitARIMA = forecast::auto.arima(y)
  if (isTRUE(PI)){
    predARIMA = forecast::forecast(fitARIMA,h=n, level = 90)
    fit_res_ARNN=forecast::nnetar(fitARIMA$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n, PI = TRUE, level = 90)
    fit_arima_arnn = predARIMA$fitted+pred_res_ARNN$fitted
    pred_arima_arnn = predARIMA$mean+pred_res_ARNN$mean
    lower_arima_arnn = predARIMA$lower+pred_res_ARNN$lower
    upper_arima_arnn = predARIMA$upper+pred_res_ARNN$upper
    forecast = list("Fitted" = fit_arima_arnn,
                    "Forecast" = pred_arima_arnn,
                    "Lower Interval" = lower_arima_arnn,
                    "Upper Interval" = upper_arima_arnn)

  }else{
    predARIMA = forecast::forecast(fitARIMA,h=n)
    fit_res_ARNN=forecast::nnetar(fitARIMA$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n)
    fit_arima_arnn = predARIMA$fitted+pred_res_ARNN$fitted
    pred_arima_arnn=predARIMA$mean+pred_res_ARNN$mean
    forecast = list("Fitted" = fit_arima_arnn,
                    "Forecast" = pred_arima_arnn)

  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Hybrid ARIMA ANN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @import forecast nnfor
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of confidence interval (\code{PI} = TRUE) for the forecast.
#' @export
#'
#' @references \itemize{
#'
#' \item Zhang, G. P. (2003). Time series forecasting using a hybrid ARIMA and neural network model.
#' Neurocomputing, 50, 159-175.
#'
#' \item Chakraborty, T., Ghosh, I., Mahajan, T., & Arora, T. (2022). Nowcasting of COVID-19 confirmed
#' cases: Foundations, trends, and challenges. Modeling, Control and Drug Development for COVID-19
#' Outbreak Prevention, 1023-1064.}
#'
arima_ann = function(y, n, PI = FALSE){
  fitARIMA = forecast::auto.arima(y)
  predARIMA = forecast::forecast(fitARIMA,h=n)
  fit_res_ANN = nnfor::mlp(fitARIMA$residuals)
  pred_res_ANN = forecast::forecast(fit_res_ANN, h=n)
  pred_arima_ann = predARIMA$mean+pred_res_ANN$mean
  if (isTRUE(PI)){
    upper = pred_arima_ann + 1.5*stats::sd(pred_arima_ann)
    lower = pred_arima_ann - 1.5*stats::sd(pred_arima_ann)
    #fit_wa_ann = fit_wa$FinalPrediction+pred_wa_ANN$fitted
    forecast = list("Forecast" = pred_arima_ann,
                    "Lower Interval" = lower,
                    "Upper Interval" = upper)

  }else{
    forecast = list("Forecast" = pred_arima_ann)
  }
  return(forecast)
}


#' Hybrid WARIMA ANN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param p An integer indicating the maximum order of AR process. Default is 5.
#' @param q An integer indicating the maximum order of MA process. Default is 5.
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @import forecast stats WaveletArima nnfor
#'
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of confidence interval (\code{PI} = TRUE) for the forecast.
#'
#' @export
#'
#' @references \itemize{
#' \item Chakraborty, T., Ghosh, I., Mahajan, T., & Arora, T. (2022). Nowcasting of COVID-19 confirmed
#' cases: Foundations, trends, and challenges. Modeling, Control and Drug Development for COVID-19
#' Outbreak Prevention, 1023-1064.}
#'
#'
warima_ann = function(y, n, p = 5, q = 5, PI = FALSE){
  fit_wa <- WaveletArima::WaveletFittingarma(y, Waveletlevels = floor(log(length(y))),
                                             boundary = 'periodic', FastFlag = TRUE,
                                             MaxARParam = p, MaxMAParam = q, NForecast = n)
  res_wa = y - fit_wa$FinalPrediction
  fit_wa_ANN = nnfor::mlp(res_wa)
  if (isTRUE(PI)){
    pred_wa_ANN = forecast::forecast(fit_wa_ANN, h=n)
    pred_wa_ann = fit_wa$Finalforecast+pred_wa_ANN$mean
    upper_wa = pred_wa_ann + 1.5*stats::sd(pred_wa_ann)
    lower_wa = pred_wa_ann - 1.5*stats::sd(pred_wa_ann)
    #fit_wa_ann = fit_wa$FinalPrediction+pred_wa_ANN$fitted
    forecast = list("Forecast" = pred_wa_ann,
      "Lower Interval" = lower_wa,
      "Upper Interval" = upper_wa) #"Fitted" = fit_wa_ann,
  }else{
    pred_wa_ANN = forecast::forecast(fit_wa_ANN, h=n)
    pred_wa_ann = fit_wa$Finalforecast+pred_wa_ANN$mean
    #fit_wa_ann = fit_wa$FinalPrediction+pred_wa_ANN$fitted
    forecast = list("Forecast" = pred_wa_ann)
  }
  return(forecast)
}

#' Hybrid WARIMA ARNN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param p An integer indicating the maximum order of AR process. Default is 5.
#' @param q An integer indicating the maximum order of MA process. Default is 5.
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast stats WaveletArima
#'
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE) for the forecast.
#'
#' @export
#'
#' @references \itemize{
#' \item Chakraborty, T., Ghosh, I., Mahajan, T., & Arora, T. (2022). Nowcasting of COVID-19 confirmed
#' cases: Foundations, trends, and challenges. Modeling, Control and Drug Development for COVID-19
#' Outbreak Prevention, 1023-1064.}
#'
#' @examples
#' warima_arnn(y = datasets::lynx, n = 3)
#'
warima_arnn = function(y, n, p = 5, q = 5, PI = FALSE, ret_fit = FALSE){
  fit_wa <- WaveletArima::WaveletFittingarma(y, Waveletlevels = floor(log(length(y))),
                                             boundary = 'periodic', FastFlag = TRUE,
                                             MaxARParam = p, MaxMAParam = q, NForecast = n)
  res_wa = y - fit_wa$FinalPrediction
  fit_wa_ARNN  = forecast::nnetar(res_wa)
  if (isTRUE(PI)){
    pred_wa_ARNN = forecast::forecast(fit_wa_ARNN, h=n, PI = TRUE, level = 86)
    upper = fit_wa$Finalforecast + 1.5*stats::sd(fit_wa$Finalforecast)
    lower = fit_wa$Finalforecast - 1.5*stats::sd(fit_wa$Finalforecast)
    fit_wa_arnn = fit_wa$FinalPrediction+pred_wa_ARNN$fitted
    pred_wa_arnn = fit_wa$Finalforecast+pred_wa_ARNN$mean
    lower_wa = lower + pred_wa_ARNN$lower
    upper_wa = upper + pred_wa_ARNN$upper
    forecast = list("Fitted" = fit_wa_arnn,
                    "Forecast" = pred_wa_arnn,
                    "Lower Interval" = lower_wa,
                    "Upper Interval" = upper_wa)
  }else{
    pred_wa_ARNN = forecast::forecast(fit_wa_ARNN, h=n)
    fit_wa_arnn = fit_wa$FinalPrediction+pred_wa_ARNN$fitted
    pred_wa_arnn = fit_wa$Finalforecast+pred_wa_ARNN$mean
    forecast = list("Fitted" = fit_wa_arnn,
                    "Forecast" = pred_wa_arnn)
  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Hybrid Theta ARNN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast stats
#'
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE) for the forecast.
#'
#' @export
#'
#' @references \itemize{
#' \item Bhattacharyya, A., Chakraborty, T., & Rai, S. N. (2022). Stochastic forecasting of
#' COVID-19 daily new cases across countries with a novel hybrid time series model.
#' Nonlinear Dynamics, 1-16.
#'
#' \item Bhattacharyya, A., Chattopadhyay, S., Pattnaik, M., & Chakraborty, T. (2021, July).
#' Theta Autoregressive Neural Network: A Hybrid Time Series Model for Pandemic Forecasting.
#' In 2021 International Joint Conference on Neural Networks (IJCNN) (pp. 1-8). IEEE.}
#'
#' @examples
#' theta_arnn(y = datasets::lynx, n = 3)
#'
theta_arnn = function(y, n, PI = FALSE, ret_fit = FALSE){
  fit_ta = forecast::thetaf(y, h = n, level = 90)
  pred_ta = stats::fitted(fit_ta)
  res_ta = stats::residuals(fit_ta)
  fore_tarnn  = forecast::nnetar(res_ta)
  if (isTRUE(PI)){
    pred_ta_ARNN = forecast::forecast(fore_tarnn, h=n, PI = TRUE, level = 90)
    upper = pred_ta_ARNN$upper + fit_ta$upper
    lower = pred_ta_ARNN$lower + fit_ta$lower
    fit_ta_arnn = pred_ta + pred_ta_ARNN$fitted
    forecast_ta_arnn = fit_ta$mean + pred_ta_ARNN$mean
    forecast = list("Fitted" = fit_ta_arnn,
                    "Forecast" = forecast_ta_arnn,
                    "Lower Interval" = lower,
                    "Upper Interval" = upper)
  }else{
    pred_ta_ARNN = forecast::forecast(fore_tarnn, h=n)
    fit_ta_arnn = pred_ta + pred_ta_ARNN$fitted
    forecast_ta_arnn = fit_ta$mean + pred_ta_ARNN$mean
    forecast = list("Fitted" = fit_ta_arnn, "Forecast" = forecast_ta_arnn)
  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Hybrid Theta ANN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @import forecast nnfor stats
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of confidence interval (\code{PI} = TRUE) for the forecast.
#' @export
#'
theta_ann = function(y, n, PI = FALSE){
  fit_tan = forecast::thetaf(y, h = n, level = 90)
  pred_tan = stats::fitted(fit_tan)
  res_tan = stats::residuals(fit_tan)
  fit_res_TANN = nnfor::mlp(res_tan)
  pred_TANN = forecast::forecast(fit_res_TANN, h=n)
  pred_theta_ann = fit_tan$mean+pred_TANN$mean
  if (isTRUE(PI)){
    upper = pred_theta_ann + 1.5*stats::sd(pred_theta_ann)
    lower = pred_theta_ann - 1.5*stats::sd(pred_theta_ann)
    #fit_wa_ann = fit_wa$FinalPrediction+pred_wa_ANN$fitted
    forecast = list("Forecast" = pred_theta_ann,
                    "Lower Interval" = lower,
                    "Upper Interval" = upper)

  }else{
    forecast = list("Forecast" = pred_theta_ann)
  }
  return(forecast)
}

#' Hybrid Random Walk ARNN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE)
#' for the forecast.
#'
#' @export
#'
#' @examples
#' rw_arnn(y = datasets::lynx, n = 3)
#'
rw_arnn = function(y, n, PI = FALSE, ret_fit = FALSE){
  fitrw = forecast::rwf(y, h = n, level = 90)
  if (isTRUE(PI)){
    fit_res_ARNN=forecast::nnetar(fitrw$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n, PI = TRUE, level = 90)
    fit_rw_arnn = fitrw$fitted+pred_res_ARNN$fitted
    pred_rw_arnn = fitrw$mean+pred_res_ARNN$mean
    lower_rw_arnn = fitrw$lower+pred_res_ARNN$lower
    upper_rw_arnn = fitrw$upper+pred_res_ARNN$upper
    forecast = list("Fitted" = fit_rw_arnn,
                    "Forecast" = pred_rw_arnn,
                    "Lower Interval" = lower_rw_arnn,
                    "Upper Interval" = upper_rw_arnn)

  }else{
    fit_res_ARNN=forecast::nnetar(fitrw$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n)
    fit_rw_arnn = fitrw$fitted+pred_res_ARNN$fitted
    pred_rw_arnn=fitrw$mean+pred_res_ARNN$mean
    forecast = list("Fitted" = fit_rw_arnn,
                    "Forecast" = pred_rw_arnn)

  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Hybrid Random Walk ANN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @import forecast nnfor
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of confidence interval (\code{PI} = TRUE) for the forecast.
#' @export
#'
rw_ann = function(y, n, PI = FALSE){
  fitrw = forecast::rwf(y, h = n)
  fit_res_ANN = nnfor::mlp(na.omit(fitrw$residuals))
  pred_res_ANN = forecast::forecast(fit_res_ANN, h=n)
  pred_rw_ann = fitrw$mean+pred_res_ANN$mean
  if (isTRUE(PI)){
    upper = pred_rw_ann + 1.5*stats::sd(pred_rw_ann)
    lower = pred_rw_ann - 1.5*stats::sd(pred_rw_ann)
    #fit_wa_ann = fit_wa$FinalPrediction+pred_wa_ANN$fitted
    forecast = list("Forecast" = pred_rw_ann,
                    "Lower Interval" = lower,
                    "Upper Interval" = upper)

  }else{
    forecast = list("Forecast" = pred_rw_ann)
  }
  return(forecast)
}

#' Hybrid ETS ARNN Forecasting Model
#'
#' @param y A numeric vector or time series
#' @param n An integer specifying the forecast horizon
#' @param PI A logical flag (default = \code{FALSE}) for generating the prediction interval.
#' @param ret_fit A logical flag specifying that the fitted values of the model on the
#' training set should be returned if true, otherwise, false (default)
#' @import forecast
#' @return The forecast of the time series of size \code{n} is generated along with the optional
#' output of fitted values (\code{ret_fit} = TRUE) and confidence interval (\code{PI} = TRUE)
#' for the forecast.
#'
#' @export
#'
#' @examples
#' ets_arnn(y = datasets::lynx, n = 3)
#'
ets_arnn = function(y, n, PI = FALSE, ret_fit = FALSE){
  fitETS = forecast::ets(y)
  if (isTRUE(PI)){
    predETS = forecast::forecast(fitETS,h=n, level = 90)
    fit_res_ARNN=forecast::nnetar(fitETS$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n, PI = TRUE, level = 90)
    fit_ets_arnn = predETS$fitted+pred_res_ARNN$fitted
    pred_ets_arnn = predETS$mean+pred_res_ARNN$mean
    lower_ets_arnn = predETS$lower+pred_res_ARNN$lower
    upper_ets_arnn = predETS$upper+pred_res_ARNN$upper
    forecast = list("Fitted" = fit_ets_arnn,
                    "Forecast" = pred_ets_arnn,
                    "Lower Interval" = lower_ets_arnn,
                    "Upper Interval" = upper_ets_arnn)

  }else{
    predETS = forecast::forecast(fitETS,h=n)
    fit_res_ARNN=forecast::nnetar(fitETS$residuals)
    pred_res_ARNN = forecast::forecast(fit_res_ARNN, h=n)
    fit_ets_arnn = predETS$fitted+pred_res_ARNN$fitted
    pred_ets_arnn=predETS$mean+pred_res_ARNN$mean
    forecast = list("Fitted" = fit_ets_arnn,
                    "Forecast" = pred_ets_arnn)

  }
  if(isTRUE(ret_fit)){
    forecast = forecast
  }else{
    forecast = forecast[-1]
  }
  return(forecast)
}

#' Summarized score of all the hybrid models implemented in this package
#'
#' @param train A numeric vector or time series object for training the hybrid models
#' @param test A numeric vector or time series object for evaluating the hybrid models
#'
#' @importFrom Metrics mase
#' @importFrom Metrics rmse
#' @importFrom Metrics smape
#' @return A data frame where the rows represent the out-of-sample scores for each of
#' the hybrid models and the columns represent the RMSE, MASE, and sMAPE scores.
#' @export
#'

summary_hybridts = function(train, test){
  metric = function(true, predict, model){
    smape = Metrics::smape(true, predict)
    mase = Metrics::mase(true, predict)
    rmse = Metrics::rmse(true, predict)
    result = data.frame("Model" = model, "RMSE" = rmse, "MASE" = mase, "SMAPE" = smape)
    return(result)
  }
  h = length(test)
  scores = data.frame()
  output = rw_ann(train, n = h)
  scr = metric(test, output$Forecast, model = "RW ANN")
  scores = rbind(scores, scr)

  output = rw_arnn(train, n = h)
  scr = metric(test, output$Forecast, model = "RW ARNN")
  scores = rbind(scores, scr)

  output = ets_arnn(train, n = h)
  scr = metric(test, output$Forecast, model = "ETS ARNN")
  scores = rbind(scores, scr)

  output = arima_warima(train, n = h)
  scr = metric(test, output$Forecast, model = "ARIMA WARIMA")
  scores = rbind(scores, scr)

  output = arima_ann(train, n = h)
  scr = metric(test, output$Forecast, model = "ARIMA ANN")
  scores = rbind(scores, scr)

  output = arima_arnn(train, n = h)
  scr = metric(test, output$Forecast, model = "ARIMA ARNN")
  scores = rbind(scores, scr)

  output = warima_ann(train, n = h)
  scr = metric(test, output$Forecast, model = "WARIMA ANN")
  scores = rbind(scores, scr)

  output = warima_arnn(train, n = h)
  scr = metric(test, output$Forecast, model = "WARIMA ARNN")
  scores = rbind(scores, scr)

  output = theta_ann(train, n = h)
  scr = metric(test, output$Forecast, model = "THETA ANN")
  scores = rbind(scores, scr)

  output = theta_arnn(train, n = h)
  scr = metric(test, output$Forecast, model = "THETA ARNN")
  scores = rbind(scores, scr)
  return(scores)
}


