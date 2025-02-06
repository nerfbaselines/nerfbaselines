#include <iostream>
#include <memory>
#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <json/json.hpp>
#include <pybind11_json/pybind11_json.hpp>
#include "src/ExpRunner.h"

namespace py = pybind11;


class CustomExpRunner : public ExpRunner {
public:
    py::dict TrainIteration(int iter_step_);
}

py::dict CustomExpRunner::TrainIteration(int iter_step_) {
  global_data_pool_->mode_ = RunningMode::TRAIN;
  global_data_pool_->iter_step_ = iter_step_;
  global_data_pool_->backward_nan_ = false;
      // global_data_pool_->drop_out_prob_ = 1.f - std::min(1.f, float(iter_step_) / 1000.f);
      // global_data_pool_->drop_out_prob_ = 0.f;

  int cur_batch_size = int(pts_batch_size_ / global_data_pool_->meaningful_sampled_pts_per_ray_) >> 4 << 4;
  auto [train_rays, gt_colors, emb_idx] = dataset_->RandRaysData(cur_batch_size, DATA_TRAIN_SET);

  Tensor& rays_o = train_rays.origins;
  Tensor& rays_d = train_rays.dirs;
  Tensor& bounds = train_rays.bounds;

  auto render_result = renderer_->Render(rays_o, rays_d, bounds, emb_idx);
  Tensor pred_colors = render_result.colors.index({Slc(0, cur_batch_size)});
  Tensor disparity = render_result.disparity;
  Tensor color_loss = torch::sqrt((pred_colors - gt_colors).square() + 1e-4f).mean();

  Tensor disparity_loss = disparity.square().mean();

  Tensor edge_feats = render_result.edge_feats;
  Tensor tv_loss = (edge_feats.index({Slc(), 0}) - edge_feats.index({Slc(), 1})).square().mean();

  Tensor sampled_weights = render_result.weights;
  Tensor idx_start_end = render_result.idx_start_end;
  Tensor sampled_var = CustomOps::WeightVar(sampled_weights, idx_start_end);
  Tensor var_loss = (sampled_var + 1e-2).sqrt().mean();

  float var_loss_weight = 0.f;
  if (iter_step_ > var_loss_end_) {
    var_loss_weight = var_loss_weight_;
  }
  else if (iter_step_ > var_loss_start_) {
    var_loss_weight = float(iter_step_ - var_loss_start_) / float(var_loss_end_ - var_loss_start_) * var_loss_weight_;
  }

  Tensor loss = color_loss + var_loss * var_loss_weight +
                disparity_loss * disp_loss_weight_ +
                tv_loss * tv_loss_weight_;

  float mse = (pred_colors - gt_colors).square().mean().item<float>();
  float psnr = 20.f * std::log10(1 / std::sqrt(mse));
  CHECK(!std::isnan(pred_colors.mean().item<float>()));
  CHECK(!std::isnan(gt_colors.mean().item<float>()));
  CHECK(!std::isnan(mse));

  py::dict out(
    "loss"_a=loss.item<float>(),
    "psnr"_a=psnr,
    "mse"_a=mse,
    "loss_color"_a=color_loss.item<float>(),
    "loss_var"_a=var_loss.item<float>(),
    "loss_disp"_a=disparity_loss.item<float>(),
    "loss_tv"_a=tv_loss.item<float>(),
    "lr"_a=optimizer_->param_groups()[0].options().get_lr()
  );

  // There can be some cases that the output colors have no grad due to the occupancy grid.
  if (loss.requires_grad()) {
    optimizer_->zero_grad();
    loss.backward();
    if (global_data_pool_->backward_nan_) {
      std::cout << "Nan!" << std::endl;
      return out;
    }
    else {
      optimizer_->step();
    }
  }

  global_data_pool_->iter_step_ = iter_step_;

  if (iter_step_ % report_freq_ == 0) {
    std::cout << fmt::format(
        "Iter: {:>6d} PSNR: {:.2f} NRays: {:>5d} OctSamples: {:.1f} Samples: {:.1f} MeaningfulSamples: {:.1f} LR: {:.4f}",
        iter_step_,
        psnr,
        cur_batch_size,
        global_data_pool_->sampled_oct_per_ray_,
        global_data_pool_->sampled_pts_per_ray_,
        global_data_pool_->meaningful_sampled_pts_per_ray_,
        optimizer_->param_groups()[0].options().get_lr())
              << std::endl;
  }
  UpdateAdaParams();
  return out;
}


PYBIND11_MODULE(f2nerf, m) {
  // Add class ExpRunner
  py::class_<CustomExpRunner>(m, "ExpRunner")
    .def(py::init<std::string>())
    .def("Train", &CustomExpRunner::Train)
    .def("TrainIteration", &CustomExpRunner::TrainIteration)
    .def("TestImages", &CustomExpRunner::TestImages)
    .def("RenderAllImages", &CustomExpRunner::RenderAllImages)
    .def("LoadCheckpoint", &CustomExpRunner::LoadCheckpoint)
    .def("SaveCheckpoint", &CustomExpRunner::SaveCheckpoint)
    .def("UpdateAdaParams", &CustomExpRunner::UpdateAdaParams)
    .def("RenderWholeImage", &CustomExpRunner::RenderWholeImage)
    .def("RenderPath", &CustomExpRunner::RenderPath)
    .def("VisualizeImage", &CustomExpRunner::VisualizeImage);
}
