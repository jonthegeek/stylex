#' Calculate Content Loss
#'
#' Content loss is calculated by comparing tensor output from the target content
#' layer to the tensor output from the generated layer.
#'
#' @param content_layer Features extracted from a specific layer of the content
#'   image.
#' @param generated_layer Features extracted from the same layer of the
#'   generated image. The tensor must be the same shape as the content image. A
#'   common cause for error is the content image and the generated image having
#'   different height and/or width.
#'
#' @return A length-1 object inheriting from class "torch_tensor" with the loss
#'   between the generated layer and the content layer.
#' @keywords internal
.calculate_content_loss <- function(content_layer, generated_layer) {
  return(
    torch::nnf_mse_loss(content_layer, generated_layer)
  )
}

#' Calculate Style Loss
#'
#' Style loss is calculated by weighting loss from multiple layers. For each
#' chosen layer, the loss measures the relative activation of the channels
#' making up that layer.
#'
#' @param input_style_features A list of tensors, each containing features
#'   extracted from a single layer for which style matters.
#' @param generated_style_features A list of tensors, the same length as
#'   input_style_features, with the equivalent features from the generated
#'   image.
#' @param style_lambdas The weights associated with each layer.
#'
#' @return A length-1 object inheriting from class "torch_tensor" with the loss
#'   between the generated layers and the style layers.
#' @keywords internal
.calculate_style_loss <- function(input_style_features,
                                  generated_style_features,
                                  style_lambdas) {
  purrr::reduce(
    purrr::pmap(
      list(
        input_style_features,
        generated_style_features,
        style_lambdas
      ),
      .weight_style_layer_loss
    ),
    `+`
  )
}

#' Calculate the Weighted Style Layer Loss
#'
#' A simple convenience function to multiple loss by weight.
#'
#' @inheritParams .calculate_style_layer_loss
#' @param lambda The weight to apply to this loss.
#'
#' @return A scalar torch_tensor representing the weighted loss for this layer.
#' @keywords internal
.weight_style_layer_loss <- function(style_layer, generated_layer, lambda) {
  return(
    lambda * .calculate_style_layer_loss(style_layer, generated_layer)
  )
}

#' Calculate Style Layer Loss
#'
#' The style loss for each layer is calculated by comparing the Gram matrix for
#' that style layer to the Gram matrix for the equivalent generated layer. The
#' intention is for this to act as a measurement that the different channels
#' that make up a layer are activated to the same proportion between the style
#' layer and the generated layer. For example, if this layer "cares" that all
#' vertical lines in the style layer are orange, measure how untrue that is for
#' the generated layer.
#'
#' @param style_layer Features extracted from a single style layer.
#' @param generated_layer Features extracted from a single generated layer.
#'   Should be the same layer as style_layer.
#'
#' @return A scalar torch_tensor representing the loss for this layer.
#' @keywords internal
.calculate_style_layer_loss <- function(style_layer, generated_layer) {
  style_gram <- .generate_gram_matrix(style_layer)
  generated_gram <- .generate_gram_matrix(generated_layer)

  return(
    torch::nnf_mse_loss(style_gram, generated_gram)
  )
}

#' Generate Gram Matrix
#'
#' Gram matrices can be used to determine how much different channels are
#' dependent on one another. This dependence is used as a definition of "style."
#'
#' @param tensor A 1 x C x H x W tensor.
#'
#' @return A tensor with shape C x C.
#' @keywords internal
.generate_gram_matrix <- function(tensor) {
  tensor_shape <- tensor$shape

  # We should only have one batch (the first dimension) in the input.
  if (tensor_shape[[1]] != 1 | length(tensor_shape) != 4) {
    rlang::abort(
      message = "The input tensor should have shape 1 x N x N x N.",
      class = "tensor_shape_error"
    )
  }

  # We only care about the channels, the height of each channel, and the width
  # of each channel.
  channels <- tensor_shape[[2]]
  height <- tensor_shape[[3]]
  width <- tensor_shape[[4]]
  area <- height * width

  tensor <- torch::torch_reshape(tensor, c(channels, area))

  # Now we multiply that reshaped tensor by its transpose, and then divide by
  # the "area" to scale.
  return(
    torch::torch_matmul(tensor, tensor$t()) / (area)
  )
}
