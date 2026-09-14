#include <mutex>
#include <string>

#include <gz/msgs/boolean.pb.h>
#include <gz/msgs/empty.pb.h>
#include <gz/msgs/stringmsg.pb.h>
#include <gz/plugin/Register.hh>
#include <gz/sim/EntityComponentManager.hh>
#include <gz/sim/System.hh>
#include <gz/sim/components/DetachableJoint.hh>
#include <gz/sim/components/Link.hh>
#include <gz/sim/components/Model.hh>
#include <gz/sim/components/Name.hh>
#include <gz/sim/components/ParentEntity.hh>
#include <gz/transport/Node.hh>

namespace erc_bringup
{
class GraspStabilizer final : public gz::sim::System,
                              public gz::sim::ISystemConfigure,
                              public gz::sim::ISystemPreUpdate
{
public:
  void Configure(const gz::sim::Entity &, const std::shared_ptr<const sdf::Element> &,
                 gz::sim::EntityComponentManager &, gz::sim::EventManager &) override
  {
    this->node.Subscribe("/grasp_stabilizer/attach", &GraspStabilizer::OnAttach, this);
    this->node.Subscribe("/grasp_stabilizer/detach", &GraspStabilizer::OnDetach, this);
    this->statePublisher = this->node.Advertise<gz::msgs::Boolean>("/grasp_stabilizer/state");
  }

  void PreUpdate(const gz::sim::UpdateInfo &,
                 gz::sim::EntityComponentManager &ecm) override
  {
    std::string target;
    bool detach = false;
    {
      std::lock_guard<std::mutex> lock(this->mutex);
      target.swap(this->pendingTarget);
      detach = this->pendingDetach;
      this->pendingDetach = false;
    }
    if (detach && this->joint != gz::sim::kNullEntity)
    {
      ecm.RequestRemoveEntity(this->joint);
      this->joint = gz::sim::kNullEntity;
      this->PublishState(false);
    }
    if (target.empty() || this->joint != gz::sim::kNullEntity)
      return;

    const auto robot = ecm.EntityByComponents(gz::sim::components::Model(),
                                               gz::sim::components::Name("tiago_pro"));
    const auto book = ecm.EntityByComponents(gz::sim::components::Model(),
                                              gz::sim::components::Name(target));
    if (robot == gz::sim::kNullEntity || book == gz::sim::kNullEntity)
      return;
    const auto gripper = ecm.EntityByComponents(
      gz::sim::components::Link(),
      gz::sim::components::Name("gripper_left_grasping_link"),
      gz::sim::components::ParentEntity(robot));
    const auto bookLink = ecm.EntityByComponents(
      gz::sim::components::Link(), gz::sim::components::Name("book_base_link"),
      gz::sim::components::ParentEntity(book));
    if (gripper == gz::sim::kNullEntity || bookLink == gz::sim::kNullEntity)
      return;
    this->joint = ecm.CreateEntity();
    ecm.CreateComponent(this->joint, gz::sim::components::DetachableJoint(
      {gripper, bookLink, "fixed"}));
    this->PublishState(true);
  }

private:
  void OnAttach(const gz::msgs::StringMsg &message)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->pendingTarget = message.data();
  }

  void OnDetach(const gz::msgs::Empty &)
  {
    std::lock_guard<std::mutex> lock(this->mutex);
    this->pendingDetach = true;
  }

  void PublishState(bool attached)
  {
    gz::msgs::Boolean message;
    message.set_data(attached);
    this->statePublisher.Publish(message);
  }

  std::mutex mutex;
  std::string pendingTarget;
  bool pendingDetach{false};
  gz::sim::Entity joint{gz::sim::kNullEntity};
  gz::transport::Node node;
  gz::transport::Node::Publisher statePublisher;
};
}

GZ_ADD_PLUGIN(erc_bringup::GraspStabilizer,
              gz::sim::System,
              gz::sim::ISystemConfigure,
              gz::sim::ISystemPreUpdate)

GZ_ADD_PLUGIN_ALIAS(erc_bringup::GraspStabilizer,
                    "erc_bringup::GraspStabilizer")
