import { Modal } from "@mui/material"

const NoCloseModal = ({
    openModal,
    children
}) => {
  return (
    <Modal
        open={openModal}
        aria-labelledby="no-close-modal-title"
        aria-describedby="no-close-modal-description"
        onClose={() => {}}
        sx={{
            borderRadius: 5
        }}
    >
      {children}
    </Modal>
  )
}

export default NoCloseModal
